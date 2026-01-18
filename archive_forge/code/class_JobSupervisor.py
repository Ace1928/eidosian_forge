import asyncio
import copy
import json
import logging
import os
import psutil
import random
import signal
import string
import subprocess
import sys
import time
import traceback
from asyncio.tasks import FIRST_COMPLETED
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from ray.util.scheduling_strategies import (
import ray
from ray._private.gcs_utils import GcsAioClient
from ray._private.utils import run_background_task
import ray._private.ray_constants as ray_constants
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
from ray.actor import ActorHandle
from ray.dashboard.consts import (
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.utils import file_tail_iterator
from ray.exceptions import ActorUnschedulableError, RuntimeEnvSetupError
from ray.job_submission import JobStatus
from ray._private.event.event_logger import get_event_logger
from ray.core.generated.event_pb2 import Event
class JobSupervisor:
    """
    Ray actor created by JobManager for each submitted job, responsible to
    setup runtime_env, execute given shell command in subprocess, update job
    status, persist job logs and manage subprocess group cleaning.

    One job supervisor actor maps to one subprocess, for one job_id.
    Job supervisor actor should fate share with subprocess it created.
    """
    DEFAULT_RAY_JOB_STOP_WAIT_TIME_S = 3
    SUBPROCESS_POLL_PERIOD_S = 0.1
    VALID_STOP_SIGNALS = ['SIGINT', 'SIGTERM']

    def __init__(self, job_id: str, entrypoint: str, user_metadata: Dict[str, str], gcs_address: str):
        self._job_id = job_id
        gcs_aio_client = GcsAioClient(address=gcs_address)
        self._job_info_client = JobInfoStorageClient(gcs_aio_client)
        self._log_client = JobLogStorageClient()
        self._entrypoint = entrypoint
        self._metadata = {JOB_ID_METADATA_KEY: job_id, JOB_NAME_METADATA_KEY: job_id}
        self._metadata.update(user_metadata)
        self._stop_event = asyncio.Event()
        self._win32_job_object = None

    def _get_driver_runtime_env(self, resources_specified: bool=False) -> Dict[str, Any]:
        """Get the runtime env that should be set in the job driver.

        Args:
            resources_specified: Whether the user specified resources (CPUs, GPUs,
                custom resources) in the submit_job request. If so, we will skip
                the workaround for GPU detection introduced in #24546, so that the
                behavior matches that of the user specifying resources for any
                other actor.

        Returns:
            The runtime env that should be set in the job driver.
        """
        curr_runtime_env = dict(ray.get_runtime_context().runtime_env)
        if resources_specified:
            return curr_runtime_env
        env_vars = curr_runtime_env.get('env_vars', {})
        env_vars.pop(ray_constants.NOSET_CUDA_VISIBLE_DEVICES_ENV_VAR)
        env_vars.pop(ray_constants.RAY_WORKER_NICENESS)
        curr_runtime_env['env_vars'] = env_vars
        return curr_runtime_env

    def ping(self):
        """Used to check the health of the actor."""
        pass

    def _exec_entrypoint(self, logs_path: str) -> subprocess.Popen:
        """
        Runs the entrypoint command as a child process, streaming stderr &
        stdout to given log files.

        Unix systems:
        Meanwhile we start a demon process and group driver
        subprocess in same pgid, such that if job actor dies, entire process
        group also fate share with it.

        Windows systems:
        A jobObject is created to enable fate sharing for the entire process group.

        Args:
            logs_path: File path on head node's local disk to store driver
                command's stdout & stderr.
        Returns:
            child_process: Child process that runs the driver command. Can be
                terminated or killed upon user calling stop().
        """
        with open(logs_path, 'w') as logs_file:
            child_process = subprocess.Popen(self._entrypoint, shell=True, start_new_session=True, stdout=logs_file, stderr=subprocess.STDOUT, preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT}) if sys.platform != 'win32' and os.environ.get('RAY_JOB_STOP_SIGNAL') == 'SIGINT' else None)
            parent_pid = os.getpid()
            child_pid = child_process.pid
            if sys.platform != 'win32':
                try:
                    child_pgid = os.getpgid(child_pid)
                except ProcessLookupError:
                    return child_process
                subprocess.Popen(f'while kill -s 0 {parent_pid}; do sleep 1; done; kill -9 -{child_pgid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys.platform == 'win32' and win32api:
                self._win32_job_object = win32job.CreateJobObject(None, '')
                win32_job_info = win32job.QueryInformationJobObject(self._win32_job_object, win32job.JobObjectExtendedLimitInformation)
                win32_job_info['BasicLimitInformation']['LimitFlags'] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                win32job.SetInformationJobObject(self._win32_job_object, win32job.JobObjectExtendedLimitInformation, win32_job_info)
                child_handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE | win32con.PROCESS_SET_QUOTA, False, child_pid)
                win32job.AssignProcessToJobObject(self._win32_job_object, child_handle)
            return child_process

    def _get_driver_env_vars(self, resources_specified: bool) -> Dict[str, str]:
        """Returns environment variables that should be set in the driver."""
        if ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE in os.environ:
            os.environ[ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE] = 'auto'
        ray_addr = ray._private.services.canonicalize_bootstrap_address_or_die('auto', ray.worker._global_node._ray_params.temp_dir)
        assert ray_addr is not None
        return {RAY_JOB_CONFIG_JSON_ENV_VAR: json.dumps({'runtime_env': self._get_driver_runtime_env(resources_specified), 'metadata': self._metadata}), ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE: ray_addr, 'PYTHONUNBUFFERED': '1'}

    async def _polling(self, child_process: subprocess.Popen) -> int:
        while child_process is not None:
            return_code = child_process.poll()
            if return_code is not None:
                return return_code
            else:
                await asyncio.sleep(self.SUBPROCESS_POLL_PERIOD_S)

    async def _poll_all(self, processes: List[psutil.Process]):
        """Poll processes until all are completed."""
        while True:
            _, alive = psutil.wait_procs(processes, timeout=0)
            if len(alive) == 0:
                return
            else:
                await asyncio.sleep(self.SUBPROCESS_POLL_PERIOD_S)

    def _kill_processes(self, processes: List[psutil.Process], sig: signal.Signals):
        """Ensure each process is already finished or send a kill signal."""
        for proc in processes:
            try:
                os.kill(proc.pid, sig)
            except ProcessLookupError:
                pass

    async def run(self, _start_signal_actor: Optional[ActorHandle]=None, resources_specified: bool=False):
        """
        Stop and start both happen asynchronously, coordinated by asyncio event
        and coroutine, respectively.

        1) Sets job status as running
        2) Pass runtime env and metadata to subprocess as serialized env
            variables.
        3) Handle concurrent events of driver execution and
        """
        curr_info = await self._job_info_client.get_info(self._job_id)
        if curr_info is None:
            raise RuntimeError(f'Status could not be retrieved for job {self._job_id}.')
        curr_status = curr_info.status
        curr_message = curr_info.message
        if curr_status == JobStatus.RUNNING:
            raise RuntimeError(f'Job {self._job_id} is already in RUNNING state. JobSupervisor.run() should only be called once. ')
        if curr_status != JobStatus.PENDING:
            raise RuntimeError(f'Job {self._job_id} is not in PENDING state. Current status is {curr_status} with message {curr_message}.')
        if _start_signal_actor:
            await _start_signal_actor.wait.remote()
        driver_agent_http_address = f'http://{ray.worker.global_worker.node.node_ip_address}:{ray.worker.global_worker.node.dashboard_agent_listen_port}'
        driver_node_id = ray.worker.global_worker.current_node_id.hex()
        await self._job_info_client.put_status(self._job_id, JobStatus.RUNNING, jobinfo_replace_kwargs={'driver_agent_http_address': driver_agent_http_address, 'driver_node_id': driver_node_id})
        try:
            os.environ.update(self._get_driver_env_vars(resources_specified))
            logger.info(f'Submitting job with RAY_ADDRESS = {os.environ[ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE]}')
            log_path = self._log_client.get_log_file_path(self._job_id)
            child_process = self._exec_entrypoint(log_path)
            child_pid = child_process.pid
            polling_task = create_task(self._polling(child_process))
            finished, _ = await asyncio.wait([polling_task, create_task(self._stop_event.wait())], return_when=FIRST_COMPLETED)
            if self._stop_event.is_set():
                polling_task.cancel()
                if sys.platform == 'win32' and self._win32_job_object:
                    win32job.TerminateJobObject(self._win32_job_object, -1)
                elif sys.platform != 'win32':
                    stop_signal = os.environ.get('RAY_JOB_STOP_SIGNAL', 'SIGTERM')
                    if stop_signal not in self.VALID_STOP_SIGNALS:
                        logger.warning(f'{stop_signal} not a valid stop signal. Terminating job with SIGTERM.')
                        stop_signal = 'SIGTERM'
                    job_process = psutil.Process(child_pid)
                    proc_to_kill = [job_process] + job_process.children(recursive=True)
                    self._kill_processes(proc_to_kill, getattr(signal, stop_signal))
                    try:
                        stop_job_wait_time = int(os.environ.get('RAY_JOB_STOP_WAIT_TIME_S', self.DEFAULT_RAY_JOB_STOP_WAIT_TIME_S))
                        poll_job_stop_task = create_task(self._poll_all(proc_to_kill))
                        await asyncio.wait_for(poll_job_stop_task, stop_job_wait_time)
                        logger.info(f'Job {self._job_id} has been terminated gracefully with {stop_signal}.')
                    except asyncio.TimeoutError:
                        logger.warning(f'Attempt to gracefully terminate job {self._job_id} through {stop_signal} has timed out after {stop_job_wait_time} seconds. Job is now being force-killed with SIGKILL.')
                        self._kill_processes(proc_to_kill, signal.SIGKILL)
                await self._job_info_client.put_status(self._job_id, JobStatus.STOPPED)
            else:
                assert len(finished) == 1, 'Should have only one coroutine done'
                [child_process_task] = finished
                return_code = child_process_task.result()
                logger.info(f'Job {self._job_id} entrypoint command exited with code {return_code}')
                if return_code == 0:
                    await self._job_info_client.put_status(self._job_id, JobStatus.SUCCEEDED, driver_exit_code=return_code)
                else:
                    log_tail = self._log_client.get_last_n_log_lines(self._job_id)
                    if log_tail is not None and log_tail != '':
                        message = f'Job entrypoint command failed with exit code {return_code}, last available logs (truncated to 20,000 chars):\n' + log_tail
                    else:
                        message = f'Job entrypoint command failed with exit code {return_code}. No logs available.'
                    await self._job_info_client.put_status(self._job_id, JobStatus.FAILED, message=message, driver_exit_code=return_code)
        except Exception:
            logger.error(f'Got unexpected exception while trying to execute driver command. {traceback.format_exc()}')
            try:
                await self._job_info_client.put_status(self._job_id, JobStatus.FAILED, message=traceback.format_exc())
            except Exception:
                logger.error(f'Failed to update job status to FAILED. Exception: {traceback.format_exc()}')
        finally:
            ray.actor.exit_actor()

    def stop(self):
        """Set step_event and let run() handle the rest in its asyncio.wait()."""
        self._stop_event.set()
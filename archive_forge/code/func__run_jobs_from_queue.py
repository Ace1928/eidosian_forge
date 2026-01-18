import ctypes
import logging
import os
import queue
import socket
import threading
import time
import traceback
import wandb
from wandb import wandb_sdk
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps import utils as sweep_utils
def _run_jobs_from_queue(self):
    global _INSTANCES
    _INSTANCES += 1
    try:
        waiting = False
        count = 0
        while True:
            if self._exit_flag:
                return
            try:
                try:
                    job = self._queue.get(timeout=5)
                    if self._exit_flag:
                        logger.debug('Exiting main loop due to exit flag.')
                        wandb.termlog('Sweep Agent: Exiting.')
                        return
                except queue.Empty:
                    if not waiting:
                        logger.debug('Paused.')
                        wandb.termlog('Sweep Agent: Waiting for job.')
                        waiting = True
                    time.sleep(5)
                    if self._exit_flag:
                        logger.debug('Exiting main loop due to exit flag.')
                        wandb.termlog('Sweep Agent: Exiting.')
                        return
                    continue
                if waiting:
                    logger.debug('Resumed.')
                    wandb.termlog('Job received.')
                    waiting = False
                count += 1
                run_id = job.run_id
                if self._run_status[run_id] == RunStatus.STOPPED:
                    continue
                logger.debug(f'Spawning new thread for run {run_id}.')
                thread = threading.Thread(target=self._run_job, args=(job,))
                self._run_threads[run_id] = thread
                thread.start()
                self._run_status[run_id] = RunStatus.RUNNING
                thread.join()
                logger.debug(f'Thread joined for run {run_id}.')
                if self._run_status[run_id] == RunStatus.RUNNING:
                    self._run_status[run_id] = RunStatus.DONE
                elif self._run_status[run_id] == RunStatus.ERRORED:
                    exc = self._exceptions[run_id]
                    exc_type, exc_value, exc_traceback = (exc.__class__, exc, exc.__traceback__)
                    exc_traceback_formatted = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    exc_repr = ''.join(exc_traceback_formatted)
                    logger.error(f'Run {run_id} errored:\n{exc_repr}')
                    wandb.termerror(f'Run {run_id} errored:\n{exc_repr}')
                    if os.getenv(wandb.env.AGENT_DISABLE_FLAPPING) == 'true':
                        self._exit_flag = True
                        return
                    elif time.time() - self._start_time < self.FLAPPING_MAX_SECONDS and len(self._exceptions) >= self.FLAPPING_MAX_FAILURES:
                        msg = 'Detected {} failed runs in the first {} seconds, killing sweep.'.format(self.FLAPPING_MAX_FAILURES, self.FLAPPING_MAX_SECONDS)
                        logger.error(msg)
                        wandb.termerror(msg)
                        wandb.termlog('To disable this check set WANDB_AGENT_DISABLE_FLAPPING=true')
                        self._exit_flag = True
                        return
                    if self._max_initial_failures < len(self._exceptions) and len(self._exceptions) >= count:
                        msg = 'Detected {} failed runs in a row at start, killing sweep.'.format(self._max_initial_failures)
                        logger.error(msg)
                        wandb.termerror(msg)
                        wandb.termlog('To change this value set WANDB_AGENT_MAX_INITIAL_FAILURES=val')
                        self._exit_flag = True
                        return
                if self._count and self._count == count:
                    logger.debug('Exiting main loop because max count reached.')
                    self._exit_flag = True
                    return
            except KeyboardInterrupt:
                logger.debug('Ctrl + C detected. Stopping sweep.')
                wandb.termlog('Ctrl + C detected. Stopping sweep.')
                self._exit()
                return
            except Exception as e:
                if self._exit_flag:
                    logger.debug('Exiting main loop due to exit flag.')
                    wandb.termlog('Sweep Agent: Killed.')
                    return
                else:
                    raise e
    finally:
        _INSTANCES -= 1
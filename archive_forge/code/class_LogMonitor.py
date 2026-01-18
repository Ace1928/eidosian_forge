import argparse
import errno
import glob
import logging
import logging.handlers
import os
import platform
import re
import shutil
import time
import traceback
from typing import Callable, List, Optional, Set
from ray._raylet import GcsClient
import ray._private.ray_constants as ray_constants
import ray._private.services as services
import ray._private.utils
from ray._private.ray_logging import setup_component_logger
class LogMonitor:
    """A monitor process for monitoring Ray log files.

    This class maintains a list of open files and a list of closed log files. We
    can't simply leave all files open because we'll run out of file
    descriptors.

    The "run" method of this class will cycle between doing several things:
    1. First, it will check if any new files have appeared in the log
       directory. If so, they will be added to the list of closed files.
    2. Then, if we are unable to open any new files, we will close all of the
       files.
    3. Then, we will open as many closed files as we can that may have new
       lines (judged by an increase in file size since the last time the file
       was opened).
    4. Then we will loop through the open files and see if there are any new
       lines in the file. If so, we will publish them to Ray pubsub.

    Attributes:
        ip: The hostname of this machine, for grouping log messages.
        logs_dir: The directory that the log files are in.
        log_filenames: This is the set of filenames of all files in
            open_file_infos and closed_file_infos.
        open_file_infos (list[LogFileInfo]): Info for all of the open files.
        closed_file_infos (list[LogFileInfo]): Info for all of the closed
            files.
        can_open_more_files: True if we can still open more files and
            false otherwise.
        max_files_open: The maximum number of files that can be open.
    """

    def __init__(self, node_ip_address: str, logs_dir: str, gcs_publisher: ray._raylet.GcsPublisher, is_proc_alive_fn: Callable[[int], bool], max_files_open: int=ray_constants.LOG_MONITOR_MAX_OPEN_FILES, gcs_address: Optional[str]=None):
        """Initialize the log monitor object."""
        self.ip: str = node_ip_address
        self.logs_dir: str = logs_dir
        self.publisher = gcs_publisher
        self.log_filenames: Set[str] = set()
        self.open_file_infos: List[LogFileInfo] = []
        self.closed_file_infos: List[LogFileInfo] = []
        self.can_open_more_files: bool = True
        self.max_files_open: int = max_files_open
        self.is_proc_alive_fn: Callable[[int], bool] = is_proc_alive_fn
        self.is_autoscaler_v2: bool = self.get_is_autoscaler_v2(gcs_address)
        logger.info(f'Starting log monitor with [max open files={max_files_open}], [is_autoscaler_v2={self.is_autoscaler_v2}]')

    def get_is_autoscaler_v2(self, gcs_address: Optional[str]) -> bool:
        """Check if autoscaler v2 is enabled."""
        if gcs_address is None:
            return False
        if not ray.experimental.internal_kv._internal_kv_initialized():
            gcs_client = GcsClient(address=gcs_address)
            ray.experimental.internal_kv._initialize_internal_kv(gcs_client)
        from ray.autoscaler.v2.utils import is_autoscaler_v2
        return is_autoscaler_v2()

    def _close_all_files(self):
        """Close all open files (so that we can open more)."""
        while len(self.open_file_infos) > 0:
            file_info = self.open_file_infos.pop(0)
            file_info.file_handle.close()
            file_info.file_handle = None
            proc_alive = True
            if file_info.worker_pid != 'raylet' and file_info.worker_pid != 'gcs_server' and (file_info.worker_pid != 'autoscaler') and (file_info.worker_pid != 'runtime_env') and (file_info.worker_pid is not None):
                assert not isinstance(file_info.worker_pid, str), f'PID should be an int type. Given PID: {file_info.worker_pid}.'
                proc_alive = self.is_proc_alive_fn(file_info.worker_pid)
                if not proc_alive:
                    target = os.path.join(self.logs_dir, 'old', os.path.basename(file_info.filename))
                    try:
                        shutil.move(file_info.filename, target)
                    except (IOError, OSError) as e:
                        if e.errno == errno.ENOENT:
                            logger.warning(f'Warning: The file {file_info.filename} was not found.')
                        else:
                            raise e
            if proc_alive:
                self.closed_file_infos.append(file_info)
        self.can_open_more_files = True

    def update_log_filenames(self):
        """Update the list of log files to monitor."""
        monitor_log_paths = []
        monitor_log_paths += glob.glob(f'{self.logs_dir}/worker*[.out|.err]') + glob.glob(f'{self.logs_dir}/java-worker*.log')
        monitor_log_paths += glob.glob(f'{self.logs_dir}/raylet*.err')
        if not self.is_autoscaler_v2:
            monitor_log_paths += glob.glob(f'{self.logs_dir}/monitor.log')
        else:
            monitor_log_paths += glob.glob(f'{self.logs_dir}/events/event_AUTOSCALER.log')
        monitor_log_paths += glob.glob(f'{self.logs_dir}/gcs_server*.err')
        if RAY_RUNTIME_ENV_LOG_TO_DRIVER_ENABLED:
            monitor_log_paths += glob.glob(f'{self.logs_dir}/runtime_env*.log')
        for file_path in monitor_log_paths:
            if os.path.isfile(file_path) and file_path not in self.log_filenames:
                worker_match = WORKER_LOG_PATTERN.match(file_path)
                if worker_match:
                    worker_pid = int(worker_match.group(2))
                else:
                    worker_pid = None
                job_id = None
                if 'runtime_env' in file_path:
                    runtime_env_job_match = RUNTIME_ENV_SETUP_PATTERN.match(file_path)
                    if runtime_env_job_match:
                        job_id = runtime_env_job_match.group(1)
                is_err_file = file_path.endswith('err')
                self.log_filenames.add(file_path)
                self.closed_file_infos.append(LogFileInfo(filename=file_path, size_when_last_opened=0, file_position=0, file_handle=None, is_err_file=is_err_file, job_id=job_id, worker_pid=worker_pid))
                log_filename = os.path.basename(file_path)
                logger.info(f'Beginning to track file {log_filename}')

    def open_closed_files(self):
        """Open some closed files if they may have new lines.

        Opening more files may require us to close some of the already open
        files.
        """
        if not self.can_open_more_files:
            self._close_all_files()
        files_with_no_updates = []
        while len(self.closed_file_infos) > 0:
            if len(self.open_file_infos) >= self.max_files_open:
                self.can_open_more_files = False
                break
            file_info = self.closed_file_infos.pop(0)
            assert file_info.file_handle is None
            try:
                file_size = os.path.getsize(file_info.filename)
            except (IOError, OSError) as e:
                if e.errno == errno.ENOENT:
                    logger.warning(f'Warning: The file {file_info.filename} was not found.')
                    self.log_filenames.remove(file_info.filename)
                    continue
                raise e
            if file_size > file_info.size_when_last_opened:
                try:
                    f = open(file_info.filename, 'rb')
                except (IOError, OSError) as e:
                    if e.errno == errno.ENOENT:
                        logger.warning(f'Warning: The file {file_info.filename} was not found.')
                        self.log_filenames.remove(file_info.filename)
                        continue
                    else:
                        raise e
                f.seek(file_info.file_position)
                file_info.size_when_last_opened = file_size
                file_info.file_handle = f
                self.open_file_infos.append(file_info)
            else:
                files_with_no_updates.append(file_info)
        if len(self.open_file_infos) >= self.max_files_open:
            self.can_open_more_files = False
        self.closed_file_infos += files_with_no_updates

    def check_log_files_and_publish_updates(self):
        """Gets updates to the log files and publishes them.

        Returns:
            True if anything was published and false otherwise.
        """
        anything_published = False
        lines_to_publish = []

        def flush():
            nonlocal lines_to_publish
            nonlocal anything_published
            if len(lines_to_publish) > 0:
                data = {'ip': self.ip, 'pid': file_info.worker_pid, 'job': file_info.job_id, 'is_err': file_info.is_err_file, 'lines': lines_to_publish, 'actor_name': file_info.actor_name, 'task_name': file_info.task_name}
                try:
                    self.publisher.publish_logs(data)
                except Exception:
                    logger.exception(f'Failed to publish log messages {data}')
                anything_published = True
                lines_to_publish = []
        for file_info in self.open_file_infos:
            assert not file_info.file_handle.closed
            file_info.reopen_if_necessary()
            max_num_lines_to_read = ray_constants.LOG_MONITOR_NUM_LINES_TO_READ
            for _ in range(max_num_lines_to_read):
                try:
                    next_line = file_info.file_handle.readline()
                    next_line = next_line.decode('utf-8', 'replace')
                    if next_line == '':
                        break
                    next_line = next_line.rstrip('\r\n')
                    if next_line.startswith(ray_constants.LOG_PREFIX_ACTOR_NAME):
                        flush()
                        file_info.actor_name = next_line.split(ray_constants.LOG_PREFIX_ACTOR_NAME, 1)[1]
                        file_info.task_name = None
                    elif next_line.startswith(ray_constants.LOG_PREFIX_TASK_NAME):
                        flush()
                        file_info.task_name = next_line.split(ray_constants.LOG_PREFIX_TASK_NAME, 1)[1]
                    elif next_line.startswith(ray_constants.LOG_PREFIX_JOB_ID):
                        file_info.job_id = next_line.split(ray_constants.LOG_PREFIX_JOB_ID, 1)[1]
                    elif next_line.startswith(ray_constants.LOG_PREFIX_TASK_ATTEMPT_START) or next_line.startswith(ray_constants.LOG_PREFIX_TASK_ATTEMPT_END):
                        pass
                    elif next_line.startswith('Windows fatal exception: access violation'):
                        file_info.file_handle.readline()
                    else:
                        lines_to_publish.append(next_line)
                except Exception:
                    logger.error(f'Error: Reading file: {file_info.filename}, position: {file_info.file_info.file_handle.tell()} failed.')
                    raise
            if file_info.file_position == 0:
                filename = file_info.filename.replace('\\', '/')
                if '/raylet' in filename:
                    file_info.worker_pid = 'raylet'
                elif '/gcs_server' in filename:
                    file_info.worker_pid = 'gcs_server'
                elif '/monitor' in filename or 'event_AUTOSCALER' in filename:
                    file_info.worker_pid = 'autoscaler'
                elif '/runtime_env' in filename:
                    file_info.worker_pid = 'runtime_env'
            file_info.file_position = file_info.file_handle.tell()
            flush()
        return anything_published

    def should_update_filenames(self, last_file_updated_time: float) -> bool:
        """Return true if filenames should be updated.

        This method is used to apply the backpressure on file updates because
        that requires heavy glob operations which use lots of CPUs.

        Args:
            last_file_updated_time: The last time filenames are updated.

        Returns:
            True if filenames should be updated. False otherwise.
        """
        elapsed_seconds = float(time.time() - last_file_updated_time)
        return len(self.log_filenames) < RAY_LOG_MONITOR_MANY_FILES_THRESHOLD or elapsed_seconds > LOG_NAME_UPDATE_INTERVAL_S

    def run(self):
        """Run the log monitor.

        This will scan the file system once every LOG_NAME_UPDATE_INTERVAL_S to
        check if there are new log files to monitor. It will also publish new
        log lines.
        """
        last_updated = time.time()
        while True:
            if self.should_update_filenames(last_updated):
                self.update_log_filenames()
                last_updated = time.time()
            self.open_closed_files()
            anything_published = self.check_log_files_and_publish_updates()
            if not anything_published:
                time.sleep(0.1)
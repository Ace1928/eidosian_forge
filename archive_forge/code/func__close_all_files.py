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
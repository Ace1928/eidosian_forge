import os
import time
import mmap
import json
import fnmatch
import asyncio
import itertools
import collections
import logging.handlers
from ray._private.utils import get_or_create_event_loop
from concurrent.futures import ThreadPoolExecutor
from ray._private.utils import run_background_task
from ray.dashboard.modules.event import event_consts
from ray.dashboard.utils import async_loop_forever
def _read_monitor_file(file, pos):
    assert isinstance(file, str), f'File should be a str, but a {type(file)}({file}) found'
    fd = os.open(file, os.O_RDONLY)
    try:
        stat = os.stat(fd)
        if stat.st_size <= 0:
            return []
        fid = stat.st_ino or file
        monitor_file = monitor_files.get(fid)
        if monitor_file:
            if monitor_file.position == monitor_file.size and monitor_file.size == stat.st_size and (monitor_file.mtime == stat.st_mtime):
                logger.debug('Skip reading the file because there is no change: %s', file)
                return []
            position = monitor_file.position
        else:
            logger.info('Found new event log file: %s', file)
            position = pos
        r = _read_file(fd, position, closefd=False)
        monitor_files[r.fid] = MonitorFile(r.size, r.mtime, r.position)
        loop.call_soon_threadsafe(callback, r.lines)
    except Exception as e:
        raise Exception(f'Read event file failed: {file}') from e
    finally:
        os.close(fd)
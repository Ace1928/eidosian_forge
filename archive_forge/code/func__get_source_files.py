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
def _get_source_files(event_dir, source_types=None, event_file_filter=None):
    event_log_names = os.listdir(event_dir)
    source_files = {}
    all_source_types = set(event_consts.EVENT_SOURCE_ALL)
    for source_type in source_types or event_consts.EVENT_SOURCE_ALL:
        assert source_type in all_source_types, f'Invalid source type: {source_type}'
        files = []
        for n in event_log_names:
            if fnmatch.fnmatch(n, f'*{source_type}*'):
                f = os.path.join(event_dir, n)
                if event_file_filter is not None and (not event_file_filter(f)):
                    continue
                files.append(f)
        if files:
            source_files[source_type] = files
    return source_files
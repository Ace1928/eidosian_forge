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
def _restore_newline(event_dict):
    try:
        event_dict['message'] = event_dict['message'].replace('\\n', '\n').replace('\\r', '\n')
    except Exception:
        logger.exception('Restore newline for event failed: %s', event_dict)
    return event_dict
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
def _parse_line(event_str):
    return _restore_newline(json.loads(event_str))
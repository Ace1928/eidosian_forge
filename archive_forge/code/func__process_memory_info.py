import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
def _process_memory_info(process):
    try:
        return process.memory_info()
    except AttributeError:
        return process.get_memory_info()
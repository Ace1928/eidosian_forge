import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from pprint import pprint
from celery.platforms import signals
from celery.utils.text import WhateverIO
def _memdump(samples=10):
    S = _mem_sample
    prev = list(S) if len(S) <= samples else sample(S, samples)
    _mem_sample[:] = []
    import gc
    gc.collect()
    after_collect = mem_rss()
    return (prev, after_collect)
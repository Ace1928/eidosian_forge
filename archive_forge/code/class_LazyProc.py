from functools import lru_cache, wraps
from typing import List, Optional, Dict
from asgiref.sync import async_to_sync
from .mp_utils import multiproc, _MAX_PROCS, lazy_parallelize
class LazyProc:

    def __init__(self, name, func, start=False, daemon=False, *args, **kwargs):
        self.name = name
        self.func = func
        self.start = start
        self.args = args
        self.kwargs = kwargs
        self.daemon = daemon

    @property
    def config(self):
        return {'target': self.func, 'daemon': self.daemon, 'args': self.args, 'kwargs': self.kwargs}
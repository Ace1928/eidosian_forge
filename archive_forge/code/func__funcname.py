from __future__ import print_function, division, absolute_import
import asyncio
import concurrent.futures
import contextlib
import time
from uuid import uuid4
import weakref
from .parallel import parallel_config
from .parallel import AutoBatchingMixin, ParallelBackendBase
def _funcname(x):
    try:
        if isinstance(x, list):
            x = x[0][0]
    except Exception:
        pass
    return funcname(x)
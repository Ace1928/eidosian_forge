import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
def _chunk_and_run(self, func, iterable, chunksize=None, unpack_args=False):
    if not hasattr(iterable, '__len__'):
        iterable = list(iterable)
    if chunksize is None:
        chunksize = self._calculate_chunksize(iterable)
    iterator = iter(iterable)
    chunk_object_refs = []
    while len(chunk_object_refs) * chunksize < len(iterable):
        actor_index = len(chunk_object_refs) % len(self._actor_pool)
        chunk_object_refs.append(self._submit_chunk(func, iterator, chunksize, actor_index, unpack_args=unpack_args))
    return chunk_object_refs
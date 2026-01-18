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
class IMapIterator:
    """Base class for OrderedIMapIterator and UnorderedIMapIterator."""

    def __init__(self, pool, func, iterable, chunksize=None):
        self._pool = pool
        self._func = func
        self._next_chunk_index = 0
        self._finished_iterating = False
        self._submitted_chunks = []
        self._ready_objects = collections.deque()
        self._iterator = iter(iterable)
        if isinstance(iterable, collections.abc.Iterator):
            self._chunksize = chunksize or 1
            result_list_size = float('inf')
        else:
            self._chunksize = chunksize or pool._calculate_chunksize(iterable)
            result_list_size = div_round_up(len(iterable), chunksize)
        self._result_thread = ResultThread([], total_object_refs=result_list_size)
        self._result_thread.start()
        for _ in range(len(self._pool._actor_pool)):
            self._submit_next_chunk()

    def _submit_next_chunk(self):
        if self._finished_iterating:
            return
        actor_index = len(self._submitted_chunks) % len(self._pool._actor_pool)
        chunk_iterator = itertools.islice(self._iterator, self._chunksize)
        chunk_list = list(chunk_iterator)
        if len(chunk_list) < self._chunksize:
            self._finished_iterating = True
            if len(chunk_list) == 0:
                return
        chunk_iterator = iter(chunk_list)
        new_chunk_id = self._pool._submit_chunk(self._func, chunk_iterator, self._chunksize, actor_index)
        self._submitted_chunks.append(False)
        self._result_thread.add_object_ref(new_chunk_id)
        if self._finished_iterating:
            self._result_thread.add_object_ref(ResultThread.END_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        raise NotImplementedError
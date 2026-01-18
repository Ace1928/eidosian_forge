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
class ResultThread(threading.Thread):
    """Thread that collects results from distributed actors.

    It winds down when either:
        - A pre-specified number of objects has been processed
        - When the END_SENTINEL (submitted through self.add_object_ref())
            has been received and all objects received before that have been
            processed.

    Initialize the thread with total_object_refs = float('inf') to wait for the
    END_SENTINEL.

    Args:
        object_refs (List[RayActorObjectRefs]): ObjectRefs to Ray Actor calls.
            Thread tracks whether they are ready. More ObjectRefs may be added
            with add_object_ref (or _add_object_ref internally) until the object
            count reaches total_object_refs.
        single_result: Should be True if the thread is managing function
            with a single result (like apply_async). False if the thread is managing
            a function with a List of results.
        callback: called only once at the end of the thread
            if no results were errors. If single_result=True, and result is
            not an error, callback is invoked with the result as the only
            argument. If single_result=False, callback is invoked with
            a list of all the results as the only argument.
        error_callback: called only once on the first result
            that errors. Should take an Exception as the only argument.
            If no result errors, this callback is not called.
        total_object_refs: Number of ObjectRefs that this thread
            expects to be ready. May be more than len(object_refs) since
            more ObjectRefs can be submitted after the thread starts.
            If None, defaults to len(object_refs). If float("inf"), thread runs
            until END_SENTINEL (submitted through self.add_object_ref())
            has been received and all objects received before that have
            been processed.
    """
    END_SENTINEL = None

    def __init__(self, object_refs: list, single_result: bool=False, callback: callable=None, error_callback: callable=None, total_object_refs: Optional[int]=None):
        threading.Thread.__init__(self, daemon=True)
        self._got_error = False
        self._object_refs = []
        self._num_ready = 0
        self._results = []
        self._ready_index_queue = queue.Queue()
        self._single_result = single_result
        self._callback = callback
        self._error_callback = error_callback
        self._total_object_refs = total_object_refs or len(object_refs)
        self._indices = {}
        self._new_object_refs = queue.Queue()
        for object_ref in object_refs:
            self._add_object_ref(object_ref)

    def _add_object_ref(self, object_ref):
        self._indices[object_ref] = len(self._object_refs)
        self._object_refs.append(object_ref)
        self._results.append(None)

    def add_object_ref(self, object_ref):
        self._new_object_refs.put(object_ref)

    def run(self):
        unready = copy.copy(self._object_refs)
        aggregated_batch_results = []
        while self._num_ready < self._total_object_refs:
            while True:
                try:
                    block = len(unready) == 0
                    new_object_ref = self._new_object_refs.get(block=block)
                    if new_object_ref is self.END_SENTINEL:
                        self._total_object_refs = len(self._object_refs)
                    else:
                        self._add_object_ref(new_object_ref)
                        unready.append(new_object_ref)
                except queue.Empty:
                    break
            [ready_id], unready = ray.wait(unready, num_returns=1)
            try:
                batch = ray.get(ready_id)
            except ray.exceptions.RayError as e:
                batch = [e]
            if not self._got_error:
                for result in batch:
                    if isinstance(result, Exception):
                        self._got_error = True
                        if self._error_callback is not None:
                            self._error_callback(result)
                        break
                    else:
                        aggregated_batch_results.append(result)
            self._num_ready += 1
            self._results[self._indices[ready_id]] = batch
            self._ready_index_queue.put(self._indices[ready_id])
        if not self._got_error and self._callback is not None:
            if not self._single_result:
                self._callback(aggregated_batch_results)
            else:
                self._callback(aggregated_batch_results[0])

    def got_error(self):
        return self._got_error

    def result(self, index):
        return self._results[index]

    def results(self):
        return self._results

    def next_ready_index(self, timeout=None):
        try:
            return self._ready_index_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError
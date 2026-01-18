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
class AsyncResult:
    """An asynchronous interface to task results.

    This should not be constructed directly.
    """

    def __init__(self, chunk_object_refs, callback=None, error_callback=None, single_result=False):
        self._single_result = single_result
        self._result_thread = ResultThread(chunk_object_refs, single_result, callback, error_callback)
        self._result_thread.start()

    def wait(self, timeout=None):
        """
        Returns once the result is ready or the timeout expires (does not
        raise TimeoutError).

        Args:
            timeout: timeout in milliseconds.
        """
        self._result_thread.join(timeout)

    def get(self, timeout=None):
        self.wait(timeout)
        if self._result_thread.is_alive():
            raise TimeoutError
        results = []
        for batch in self._result_thread.results():
            for result in batch:
                if isinstance(result, PoolTaskError):
                    raise result.underlying
                elif isinstance(result, Exception):
                    raise result
            results.extend(batch)
        if self._single_result:
            return results[0]
        return results

    def ready(self):
        """
        Returns true if the result is ready, else false if the tasks are still
        running.
        """
        return not self._result_thread.is_alive()

    def successful(self):
        """
        Returns true if none of the submitted tasks errored, else false. Should
        only be called once the result is ready (can be checked using `ready`).
        """
        if not self.ready():
            raise ValueError(f'{self!r} not ready')
        return not self._result_thread.got_error()
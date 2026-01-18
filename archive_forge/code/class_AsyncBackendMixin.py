import socket
import threading
import time
from collections import deque
from queue import Empty
from time import sleep
from weakref import WeakKeyDictionary
from kombu.utils.compat import detect_environment
from celery import states
from celery.exceptions import TimeoutError
from celery.utils.threads import THREAD_TIMEOUT_MAX
class AsyncBackendMixin:
    """Mixin for backends that enables the async API."""

    def _collect_into(self, result, bucket):
        self.result_consumer.buckets[result] = bucket

    def iter_native(self, result, no_ack=True, **kwargs):
        self._ensure_not_eager()
        results = result.results
        if not results:
            raise StopIteration()
        bucket = deque()
        for node in results:
            if not hasattr(node, '_cache'):
                bucket.append(node)
            elif node._cache:
                bucket.append(node)
            else:
                self._collect_into(node, bucket)
        for _ in self._wait_for_pending(result, no_ack=no_ack, **kwargs):
            while bucket:
                node = bucket.popleft()
                if not hasattr(node, '_cache'):
                    yield (node.id, node.children)
                else:
                    yield (node.id, node._cache)
        while bucket:
            node = bucket.popleft()
            yield (node.id, node._cache)

    def add_pending_result(self, result, weak=False, start_drainer=True):
        if start_drainer:
            self.result_consumer.drainer.start()
        try:
            self._maybe_resolve_from_buffer(result)
        except Empty:
            self._add_pending_result(result.id, result, weak=weak)
        return result

    def _maybe_resolve_from_buffer(self, result):
        result._maybe_set_cache(self._pending_messages.take(result.id))

    def _add_pending_result(self, task_id, result, weak=False):
        concrete, weak_ = self._pending_results
        if task_id not in weak_ and result.id not in concrete:
            (weak_ if weak else concrete)[task_id] = result
            self.result_consumer.consume_from(task_id)

    def add_pending_results(self, results, weak=False):
        self.result_consumer.drainer.start()
        return [self.add_pending_result(result, weak=weak, start_drainer=False) for result in results]

    def remove_pending_result(self, result):
        self._remove_pending_result(result.id)
        self.on_result_fulfilled(result)
        return result

    def _remove_pending_result(self, task_id):
        for mapping in self._pending_results:
            mapping.pop(task_id, None)

    def on_result_fulfilled(self, result):
        self.result_consumer.cancel_for(result.id)

    def wait_for_pending(self, result, callback=None, propagate=True, **kwargs):
        self._ensure_not_eager()
        for _ in self._wait_for_pending(result, **kwargs):
            pass
        return result.maybe_throw(callback=callback, propagate=propagate)

    def _wait_for_pending(self, result, timeout=None, on_interval=None, on_message=None, **kwargs):
        return self.result_consumer._wait_for_pending(result, timeout=timeout, on_interval=on_interval, on_message=on_message, **kwargs)

    @property
    def is_async(self):
        return True
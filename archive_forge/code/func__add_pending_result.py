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
def _add_pending_result(self, task_id, result, weak=False):
    concrete, weak_ = self._pending_results
    if task_id not in weak_ and result.id not in concrete:
        (weak_ if weak else concrete)[task_id] = result
        self.result_consumer.consume_from(task_id)
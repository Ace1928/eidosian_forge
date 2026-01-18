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
def _remove_pending_result(self, task_id):
    for mapping in self._pending_results:
        mapping.pop(task_id, None)
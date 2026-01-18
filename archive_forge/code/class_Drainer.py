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
@register_drainer('default')
class Drainer:
    """Result draining service."""

    def __init__(self, result_consumer):
        self.result_consumer = result_consumer

    def start(self):
        pass

    def stop(self):
        pass

    def drain_events_until(self, p, timeout=None, interval=1, on_interval=None, wait=None):
        wait = wait or self.result_consumer.drain_events
        time_start = time.monotonic()
        while 1:
            if timeout and time.monotonic() - time_start >= timeout:
                raise socket.timeout()
            try:
                yield self.wait_for(p, wait, timeout=interval)
            except socket.timeout:
                pass
            if on_interval:
                on_interval()
            if p.ready:
                break

    def wait_for(self, p, wait, timeout=None):
        wait(timeout=timeout)
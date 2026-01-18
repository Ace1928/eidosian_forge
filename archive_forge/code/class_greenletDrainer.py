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
class greenletDrainer(Drainer):
    spawn = None
    _g = None
    _drain_complete_event = None

    def _create_drain_complete_event(self):
        """create new self._drain_complete_event object"""
        pass

    def _send_drain_complete_event(self):
        """raise self._drain_complete_event for wakeup .wait_for"""
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._shutdown = threading.Event()
        self._create_drain_complete_event()

    def run(self):
        self._started.set()
        while not self._stopped.is_set():
            try:
                self.result_consumer.drain_events(timeout=1)
                self._send_drain_complete_event()
                self._create_drain_complete_event()
            except socket.timeout:
                pass
        self._shutdown.set()

    def start(self):
        if not self._started.is_set():
            self._g = self.spawn(self.run)
            self._started.wait()

    def stop(self):
        self._stopped.set()
        self._send_drain_complete_event()
        self._shutdown.wait(THREAD_TIMEOUT_MAX)

    def wait_for(self, p, wait, timeout=None):
        self.start()
        if not p.ready:
            self._drain_complete_event.wait(timeout=timeout)
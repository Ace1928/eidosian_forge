from __future__ import annotations
import sys
import threading
import time
import weakref
from typing import Any, Callable, Optional
from pymongo.lock import _create_lock
class PeriodicExecutor:

    def __init__(self, interval: float, min_interval: float, target: Callable[[], bool], name: Optional[str]=None):
        """ "Run a target function periodically on a background thread.

        If the target's return value is false, the executor stops.

        :Parameters:
          - `interval`: Seconds between calls to `target`.
          - `min_interval`: Minimum seconds between calls if `wake` is
            called very often.
          - `target`: A function.
          - `name`: A name to give the underlying thread.
        """
        self._event = False
        self._interval = interval
        self._min_interval = min_interval
        self._target = target
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._name = name
        self._skip_sleep = False
        self._thread_will_exit = False
        self._lock = _create_lock()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}(name={self._name}) object at 0x{id(self):x}>'

    def open(self) -> None:
        """Start. Multiple calls have no effect.

        Not safe to call from multiple threads at once.
        """
        with self._lock:
            if self._thread_will_exit:
                try:
                    assert self._thread is not None
                    self._thread.join()
                except ReferenceError:
                    pass
            self._thread_will_exit = False
            self._stopped = False
        started: Any = False
        try:
            started = self._thread and self._thread.is_alive()
        except ReferenceError:
            pass
        if not started:
            thread = threading.Thread(target=self._run, name=self._name)
            thread.daemon = True
            self._thread = weakref.proxy(thread)
            _register_executor(self)
            try:
                thread.start()
            except RuntimeError as e:
                if 'interpreter shutdown' in str(e) or sys.is_finalizing():
                    self._thread = None
                    return
                raise

    def close(self, dummy: Any=None) -> None:
        """Stop. To restart, call open().

        The dummy parameter allows an executor's close method to be a weakref
        callback; see monitor.py.
        """
        self._stopped = True

    def join(self, timeout: Optional[int]=None) -> None:
        if self._thread is not None:
            try:
                self._thread.join(timeout)
            except (ReferenceError, RuntimeError):
                pass

    def wake(self) -> None:
        """Execute the target function soon."""
        self._event = True

    def update_interval(self, new_interval: int) -> None:
        self._interval = new_interval

    def skip_sleep(self) -> None:
        self._skip_sleep = True

    def __should_stop(self) -> bool:
        with self._lock:
            if self._stopped:
                self._thread_will_exit = True
                return True
            return False

    def _run(self) -> None:
        while not self.__should_stop():
            try:
                if not self._target():
                    self._stopped = True
                    break
            except BaseException:
                with self._lock:
                    self._stopped = True
                    self._thread_will_exit = True
                raise
            if self._skip_sleep:
                self._skip_sleep = False
            else:
                deadline = time.monotonic() + self._interval
                while not self._stopped and time.monotonic() < deadline:
                    time.sleep(self._min_interval)
                    if self._event:
                        break
            self._event = False
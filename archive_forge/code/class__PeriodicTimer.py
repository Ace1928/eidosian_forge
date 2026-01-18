import ipaddress
import random
import re
import socket
import time
import weakref
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union
class _PeriodicTimer:
    """Represent a timer that periodically runs a specified function.

    Args:
        interval:
            The interval, in seconds, between each run.
        function:
            The function to run.
    """

    class _Context:
        interval: float
        function: Callable[..., None]
        args: Tuple[Any, ...]
        kwargs: Dict[str, Any]
        stop_event: Event
    _name: Optional[str]
    _thread: Optional[Thread]
    _finalizer: Optional[weakref.finalize]
    _ctx: _Context

    def __init__(self, interval: timedelta, function: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        self._name = None
        self._ctx = self._Context()
        self._ctx.interval = interval.total_seconds()
        self._ctx.function = function
        self._ctx.args = args or ()
        self._ctx.kwargs = kwargs or {}
        self._ctx.stop_event = Event()
        self._thread = None
        self._finalizer = None

    @property
    def name(self) -> Optional[str]:
        """Get the name of the timer."""
        return self._name

    def set_name(self, name: str) -> None:
        """Set the name of the timer.

        The specified name will be assigned to the background thread and serves
        for debugging and troubleshooting purposes.
        """
        if self._thread:
            raise RuntimeError('The timer has already started.')
        self._name = name

    def start(self) -> None:
        """Start the timer."""
        if self._thread:
            raise RuntimeError('The timer has already started.')
        self._thread = Thread(target=self._run, name=self._name or 'PeriodicTimer', args=(self._ctx,), daemon=True)
        self._finalizer = weakref.finalize(self, self._stop_thread, self._thread, self._ctx.stop_event)
        self._finalizer.atexit = False
        self._thread.start()

    def cancel(self) -> None:
        """Stop the timer at the next opportunity."""
        if self._finalizer:
            self._finalizer()

    @staticmethod
    def _run(ctx) -> None:
        while not ctx.stop_event.wait(ctx.interval):
            ctx.function(*ctx.args, **ctx.kwargs)

    @staticmethod
    def _stop_thread(thread, stop_event):
        stop_event.set()
        thread.join()
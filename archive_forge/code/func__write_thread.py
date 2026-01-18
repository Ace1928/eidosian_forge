from __future__ import annotations
import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator, TextIO, cast
from .application import get_app_session, run_in_terminal
from .output import Output
def _write_thread(self) -> None:
    done = False
    while not done:
        item = self._flush_queue.get()
        if isinstance(item, _Done):
            break
        if not item:
            continue
        text = []
        text.append(item)
        while True:
            try:
                item = self._flush_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if isinstance(item, _Done):
                    done = True
                else:
                    text.append(item)
        app_loop = self._get_app_loop()
        self._write_and_flush(app_loop, ''.join(text))
        if app_loop is not None:
            time.sleep(self.sleep_between_writes)
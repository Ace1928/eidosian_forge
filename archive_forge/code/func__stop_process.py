from __future__ import annotations
import functools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from watchdog.events import EVENT_TYPE_OPENED, FileSystemEvent, PatternMatchingEventHandler
from watchdog.utils import echo
from watchdog.utils.event_debouncer import EventDebouncer
from watchdog.utils.process_watcher import ProcessWatcher
def _stop_process(self):
    with self._stopping_lock:
        if self._is_process_stopping:
            return
        self._is_process_stopping = True
    try:
        if self.process_watcher is not None:
            self.process_watcher.stop()
            self.process_watcher = None
        if self.process is not None:
            try:
                kill_process(self.process.pid, self.stop_signal)
            except OSError:
                pass
            else:
                kill_time = time.time() + self.kill_after
                while time.time() < kill_time:
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.25)
                else:
                    try:
                        kill_process(self.process.pid, 9)
                    except OSError:
                        pass
            self.process = None
    finally:
        self._is_process_stopping = False
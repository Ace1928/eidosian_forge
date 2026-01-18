import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
class StreamWrapper(RedirectBase):
    """Patches the write method of current sys.stdout/sys.stderr."""

    def __init__(self, src, cbs=()):
        super().__init__(src=src, cbs=cbs)
        self._installed = False
        self._emulator = TerminalEmulator()

    def _emulator_write(self):
        while True:
            if self._queue.empty():
                if self._stopped.is_set():
                    return
                time.sleep(0.5)
                continue
            data = []
            while not self._queue.empty():
                data.append(self._queue.get())
            if self._stopped.is_set() and sum(map(len, data)) > 100000:
                wandb.termlog('Terminal output too large. Logging without processing.')
                self.flush()
                [self.flush(line.encode('utf-8')) for line in data]
                return
            try:
                self._emulator.write(''.join(data))
            except Exception:
                pass

    def _callback(self):
        while not (self._stopped.is_set() and self._queue.empty()):
            self.flush()
            time.sleep(_MIN_CALLBACK_INTERVAL)

    def install(self):
        super().install()
        if self._installed:
            return
        stream = self.src_wrapped_stream
        old_write = stream.write
        self._prev_callback_timestamp = time.time()
        self._old_write = old_write

        def write(data):
            self._old_write(data)
            self._queue.put(data)
        stream.write = write
        self._queue = queue.Queue()
        self._stopped = threading.Event()
        self._emulator_write_thread = threading.Thread(target=self._emulator_write)
        self._emulator_write_thread.daemon = True
        self._emulator_write_thread.start()
        if not wandb.run or wandb.run._settings.mode == 'online':
            self._callback_thread = threading.Thread(target=self._callback)
            self._callback_thread.daemon = True
            self._callback_thread.start()
        self._installed = True

    def flush(self, data=None):
        if data is None:
            try:
                data = self._emulator.read().encode('utf-8')
            except Exception:
                pass
        if data:
            for cb in self.cbs:
                try:
                    cb(data)
                except Exception:
                    pass

    def uninstall(self):
        if not self._installed:
            return
        self.src_wrapped_stream.write = self._old_write
        self._stopped.set()
        self._emulator_write_thread.join(timeout=5)
        if self._emulator_write_thread.is_alive():
            wandb.termlog(f'Processing terminal output ({self.src})...')
            self._emulator_write_thread.join()
            wandb.termlog('Done.')
        self.flush()
        self._installed = False
        super().uninstall()
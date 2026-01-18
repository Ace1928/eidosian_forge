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
class Redirect(RedirectBase):
    """Redirect low level file descriptors."""

    def __init__(self, src, cbs=()):
        super().__init__(src=src, cbs=cbs)
        self._installed = False
        self._emulator = TerminalEmulator()

    def _pipe(self):
        if pty:
            r, w = pty.openpty()
        else:
            r, w = os.pipe()
        return (r, w)

    def install(self):
        super().install()
        if self._installed:
            return
        self._pipe_read_fd, self._pipe_write_fd = self._pipe()
        if os.isatty(self._pipe_read_fd):
            _WSCH.add_fd(self._pipe_read_fd)
        self._orig_src_fd = os.dup(self.src_fd)
        self._orig_src = os.fdopen(self._orig_src_fd, 'wb', 0)
        os.dup2(self._pipe_write_fd, self.src_fd)
        self._installed = True
        self._queue = queue.Queue()
        self._stopped = threading.Event()
        self._pipe_relay_thread = threading.Thread(target=self._pipe_relay)
        self._pipe_relay_thread.daemon = True
        self._pipe_relay_thread.start()
        self._emulator_write_thread = threading.Thread(target=self._emulator_write)
        self._emulator_write_thread.daemon = True
        self._emulator_write_thread.start()
        if not wandb.run or wandb.run._settings.mode == 'online':
            self._callback_thread = threading.Thread(target=self._callback)
            self._callback_thread.daemon = True
            self._callback_thread.start()

    def uninstall(self):
        if not self._installed:
            return
        self._installed = False
        time.sleep(1)
        self._stopped.set()
        os.dup2(self._orig_src_fd, self.src_fd)
        os.write(self._pipe_write_fd, _LAST_WRITE_TOKEN)
        self._pipe_relay_thread.join()
        os.close(self._pipe_read_fd)
        os.close(self._pipe_write_fd)
        t = threading.Thread(target=self.src_wrapped_stream.flush)
        t.start()
        t.join(timeout=10)
        self._emulator_write_thread.join(timeout=5)
        if self._emulator_write_thread.is_alive():
            wandb.termlog(f'Processing terminal output ({self.src})...')
            self._emulator_write_thread.join()
            wandb.termlog('Done.')
        self.flush()
        _WSCH.remove_fd(self._pipe_read_fd)
        super().uninstall()

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

    def _callback(self):
        while not self._stopped.is_set():
            self.flush()
            time.sleep(_MIN_CALLBACK_INTERVAL)

    def _pipe_relay(self):
        while True:
            try:
                brk = False
                data = os.read(self._pipe_read_fd, 4096)
                if self._stopped.is_set():
                    if _LAST_WRITE_TOKEN not in data:
                        n = len(_LAST_WRITE_TOKEN)
                        while n and data[-n:] != _LAST_WRITE_TOKEN[:n]:
                            n -= 1
                        if n:
                            data += os.read(self._pipe_read_fd, len(_LAST_WRITE_TOKEN) - n)
                    if _LAST_WRITE_TOKEN in data:
                        data = data.replace(_LAST_WRITE_TOKEN, b'')
                        brk = True
                i = self._orig_src.write(data)
                if i is not None:
                    while i < len(data):
                        i += self._orig_src.write(data[i:])
                self._queue.put(data)
                if brk:
                    return
            except OSError:
                return

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
                [self.flush(line) for line in data]
                return
            try:
                self._emulator.write(b''.join(data).decode('utf-8'))
            except Exception:
                pass
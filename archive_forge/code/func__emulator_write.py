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
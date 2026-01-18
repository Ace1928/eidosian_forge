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
class StreamRawWrapper(RedirectBase):
    """Patches the write method of current sys.stdout/sys.stderr.

    Captures data in a raw form rather than using the emulator
    """

    def __init__(self, src, cbs=()):
        super().__init__(src=src, cbs=cbs)
        self._installed = False

    def save(self):
        stream = self.src_wrapped_stream
        self._old_write = stream.write

    def install(self):
        super().install()
        if self._installed:
            return
        stream = self.src_wrapped_stream
        self._prev_callback_timestamp = time.time()

        def write(data):
            self._old_write(data)
            for cb in self.cbs:
                try:
                    cb(data)
                except Exception:
                    pass
        stream.write = write
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        self.src_wrapped_stream.write = self._old_write
        self._installed = False
        super().uninstall()
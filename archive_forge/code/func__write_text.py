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
def _write_text(self, text):
    prev_end = 0
    for match in SEP_RE.finditer(text):
        start, end = match.span()
        self._write_plain_text(text[prev_end:start])
        prev_end = end
        c = match.group()
        if c == '\n':
            self.linefeed()
        elif c == '\r':
            self.carriage_return()
        elif c == '\x08':
            self.cursor_left()
        else:
            continue
    self._write_plain_text(text[prev_end:])
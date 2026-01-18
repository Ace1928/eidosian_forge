import contextlib
import fcntl
import itertools
import multiprocessing
import os
import pty
import re
import signal
import struct
import sys
import tempfile
import termios
import time
import traceback
import types
from typing import Optional, Generator, Tuple
import typing
def _lines_in_page(self) -> int:
    original_count = self._total_lines
    try:
        for line in self._lines:
            if line is self._page_end:
                break
    except IOError:
        pass
    return self._total_lines - original_count
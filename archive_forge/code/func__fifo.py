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
@contextlib.contextmanager
def _fifo(fifo_path: str) -> Generator[str, None, None]:
    try:
        os.mkfifo(fifo_path, 384)
        yield fifo_path
    finally:
        try:
            os.unlink(fifo_path)
        except OSError:
            pass
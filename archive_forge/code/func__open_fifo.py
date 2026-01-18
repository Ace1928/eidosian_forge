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
def _open_fifo(path: Optional[str], write: bool) -> Optional[int]:
    if path is None:
        return None
    return os.open(path, os.O_WRONLY if write else os.O_RDONLY)
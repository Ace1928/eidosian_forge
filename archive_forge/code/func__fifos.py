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
def _fifos(*fifo_names: Optional[str]) -> Generator[typing.List[Optional[str]], None, None]:
    with tempfile.TemporaryDirectory() as directory:
        with contextlib.ExitStack() as stack:
            fifos = [stack.enter_context(_fifo(os.path.join(directory, name))) if name is not None else None for name in fifo_names]
            yield fifos
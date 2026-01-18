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
def _iter_lines(self) -> Generator[typing.Union[str, None], None, None]:

    def get_content(segment: str) -> Tuple[bool, typing.Union[str, None]]:
        if not segment:
            return (False, '')
        if segment == '\x1b[1m~\x1b[0m\n':
            return (False, '')
        visible = self._ctrl_chars.sub('', segment)
        if (visible.rstrip() == ':' or '(END)' in visible or 'Waiting for data...' in visible) and segment.replace('\x1b[m', '') != visible:
            return (True, self._page_end)
        elif visible.rstrip() or segment == visible:
            self._total_lines += 1
            self.env.record_output(visible)
            return (True, visible)
        return (False, '')
    while True:
        line = '\x1b[?'
        while line.lstrip(' q').startswith('\x1b[?'):
            rawline = self.env.readline()
            line = rawline.replace('\x07', '').replace('\x1b[m', '')
        before, reset, after = line.partition('\x1b[2J')
        valid, content = get_content(before)
        if valid:
            yield content
        if reset and (not (valid and content is self._page_end)):
            yield self._page_end
        valid, content = get_content(after)
        if valid:
            yield content
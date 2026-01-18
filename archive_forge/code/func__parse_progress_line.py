from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def _parse_progress_line(self, line: AnyStr) -> None:
    """Parse progress information from the given line as retrieved by git-push
        or git-fetch.

        - Lines that do not contain progress info are stored in :attr:`other_lines`.
        - Lines that seem to contain an error (i.e. start with ``error:`` or ``fatal:``)
          are stored in :attr:`error_lines`.
        """
    if isinstance(line, bytes):
        line_str = line.decode('utf-8')
    else:
        line_str = line
    self._cur_line = line_str
    if self._cur_line.startswith(('error:', 'fatal:')):
        self.error_lines.append(self._cur_line)
        return
    last_valid_index = None
    for i, c in enumerate(reversed(line_str)):
        if ord(c) < 32:
            last_valid_index = -i - 1
    if last_valid_index is not None:
        line_str = line_str[:last_valid_index]
    line_str = line_str.rstrip()
    cur_count, max_count = (None, None)
    match = self.re_op_relative.match(line_str)
    if match is None:
        match = self.re_op_absolute.match(line_str)
    if not match:
        self.line_dropped(line_str)
        self.other_lines.append(line_str)
        return
    op_code = 0
    _remote, op_name, _percent, cur_count, max_count, message = match.groups()
    if op_name == 'Counting objects':
        op_code |= self.COUNTING
    elif op_name == 'Compressing objects':
        op_code |= self.COMPRESSING
    elif op_name == 'Writing objects':
        op_code |= self.WRITING
    elif op_name == 'Receiving objects':
        op_code |= self.RECEIVING
    elif op_name == 'Resolving deltas':
        op_code |= self.RESOLVING
    elif op_name == 'Finding sources':
        op_code |= self.FINDING_SOURCES
    elif op_name == 'Checking out files':
        op_code |= self.CHECKING_OUT
    else:
        self.line_dropped(line_str)
        return
    if op_code not in self._seen_ops:
        self._seen_ops.append(op_code)
        op_code |= self.BEGIN
    if message is None:
        message = ''
    message = message.strip()
    if message.endswith(self.DONE_TOKEN):
        op_code |= self.END
        message = message[:-len(self.DONE_TOKEN)]
    message = message.strip(self.TOKEN_SEPARATOR)
    self.update(op_code, cur_count and float(cur_count), max_count and float(max_count), message)
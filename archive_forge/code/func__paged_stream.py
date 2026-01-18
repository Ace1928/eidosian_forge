import contextlib
import enum
import io
import os
import signal
import subprocess
import sys
import types
import typing
from typing import Any, Optional, Type, Dict, TextIO
from autopage import command
def _paged_stream(self) -> TextIO:
    buffer_size = 1 if self._line_buffering() else -1
    self._pager = subprocess.Popen(self._command.command(), env=self._pager_env(), bufsize=buffer_size, universal_newlines=True, encoding=self._encoding(), errors=self._errors(), stdin=subprocess.PIPE, stdout=self._pager_out_stream())
    assert self._pager.stdin is not None
    return typing.cast(TextIO, self._pager.stdin)
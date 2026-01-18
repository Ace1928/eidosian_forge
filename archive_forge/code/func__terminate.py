from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
def _terminate(self) -> None:
    """Terminate the underlying process."""
    if self.proc is None:
        return
    proc = self.proc
    self.proc = None
    if proc.stdin:
        proc.stdin.close()
    if proc.stdout:
        proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()
    try:
        if proc.poll() is not None:
            self.status = self._status_code_if_terminate or proc.poll()
            return
    except OSError as ex:
        _logger.info('Ignored error after process had died: %r', ex)
    if os is None or getattr(os, 'kill', None) is None:
        return
    try:
        proc.terminate()
        status = proc.wait()
        self.status = self._status_code_if_terminate or status
    except OSError as ex:
        _logger.info('Ignored error after process had died: %r', ex)
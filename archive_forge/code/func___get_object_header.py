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
def __get_object_header(self, cmd: 'Git.AutoInterrupt', ref: AnyStr) -> Tuple[str, str, int]:
    if cmd.stdin and cmd.stdout:
        cmd.stdin.write(self._prepare_ref(ref))
        cmd.stdin.flush()
        return self._parse_object_header(cmd.stdout.readline())
    else:
        raise ValueError('cmd stdin was empty')
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
def _prepare_ref(self, ref: AnyStr) -> bytes:
    if isinstance(ref, bytes):
        refstr: str = ref.decode('ascii')
    elif not isinstance(ref, str):
        refstr = str(ref)
    else:
        refstr = ref
    if not refstr.endswith('\n'):
        refstr += '\n'
    return refstr.encode(defenc)
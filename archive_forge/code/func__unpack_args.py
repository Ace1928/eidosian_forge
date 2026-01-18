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
@classmethod
def _unpack_args(cls, arg_list: Sequence[str]) -> List[str]:
    outlist = []
    if isinstance(arg_list, (list, tuple)):
        for arg in arg_list:
            outlist.extend(cls._unpack_args(arg))
    else:
        outlist.append(str(arg_list))
    return outlist
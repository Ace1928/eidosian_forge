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
def _get_persistent_cmd(self, attr_name: str, cmd_name: str, *args: Any, **kwargs: Any) -> 'Git.AutoInterrupt':
    cur_val = getattr(self, attr_name)
    if cur_val is not None:
        return cur_val
    options = {'istream': PIPE, 'as_process': True}
    options.update(kwargs)
    cmd = self._call_process(cmd_name, *args, **options)
    setattr(self, attr_name, cmd)
    cmd = cast('Git.AutoInterrupt', cmd)
    return cmd
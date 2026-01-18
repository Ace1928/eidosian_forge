import contextlib
import os
import platform
import shlex
import shutil
import sys
import sysconfig
import tempfile
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import (
import distutils.ccompiler
import distutils.errors
def _single_compile(obj: Any) -> None:
    try:
        src, ext = build[obj]
    except KeyError:
        return
    if not os.path.exists(obj) or self.needs_recompile(obj, src):
        compiler._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
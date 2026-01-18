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
def _add_cflags(self, flags: List[str]) -> None:
    self.extra_compile_args[:0] = flags
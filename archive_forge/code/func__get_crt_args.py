from __future__ import annotations
import os.path
import re
import subprocess
import typing as T
from .. import mesonlib
from .. import mlog
from ..arglist import CompilerArgs
from ..linkers import RSPFileSyntax
from ..mesonlib import (
from . import compilers
from .compilers import (
from .mixins.gnu import GnuCompiler
from .mixins.gnu import gnu_common_warning_args
def _get_crt_args(self, crt_val: str, buildtype: str) -> T.List[str]:
    if not self.info.is_windows():
        return []
    return self.mscrt_args[self.get_crt_val(crt_val, buildtype)]
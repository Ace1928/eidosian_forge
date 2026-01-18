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
@classmethod
def _translate_arg_to_osx(cls, arg: str) -> T.List[str]:
    args: T.List[str] = []
    if arg.startswith('-install_name'):
        args.append('-L=' + arg)
    return args
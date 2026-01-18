from __future__ import annotations
import os.path
import typing as T
from ... import coredata
from ... import mesonlib
from ...mesonlib import OptionKey
from ...mesonlib import LibType
from mesonbuild.compilers.compilers import CompileCheckMode
def _get_compile_output(self, dirname: str, mode: CompileCheckMode) -> str:
    assert mode != CompileCheckMode.PREPROCESS, 'In pre-processor mode, the output is sent to stdout and discarded'
    if mode == CompileCheckMode.LINK:
        suffix = 'js'
    else:
        suffix = 'o'
    return os.path.join(dirname, 'output.' + suffix)
from __future__ import annotations
import typing as T
import os, sys
from .. import mesonlib
from .. import mlog
from ..mesonlib import Popen_safe
import argparse
def _get_implib_dllname(impfilename: str) -> T.Tuple[T.List[str], str]:
    all_stderr = ''
    for lib in (['lib'], get_tool('llvm-lib')):
        output, e = call_tool_nowarn(lib + ['-list', impfilename])
        if output:
            return (output.split('\n')[-2:-1], None)
        all_stderr += e
    output, e = call_tool_nowarn(get_tool('dlltool') + ['-I', impfilename])
    if output:
        return ([output], None)
    all_stderr += e
    return ([], all_stderr)
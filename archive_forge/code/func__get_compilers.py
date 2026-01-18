from __future__ import annotations
from ..mesonlib import (
from ..envconfig import BinaryTable
from .. import mlog
from ..linkers import guess_win_linker, guess_nix_linker
import subprocess
import platform
import re
import shutil
import tempfile
import os
import typing as T
def _get_compilers(env: 'Environment', lang: str, for_machine: MachineChoice) -> T.Tuple[T.List[T.List[str]], T.List[str], T.Optional['ExternalProgram']]:
    """
    The list of compilers is detected in the exact same way for
    C, C++, ObjC, ObjC++, Fortran, CS so consolidate it here.
    """
    value = env.lookup_binary_entry(for_machine, lang)
    if value is not None:
        comp, ccache = BinaryTable.parse_entry(value)
        compilers = [comp]
    else:
        if not env.machines.matches_build_machine(for_machine):
            raise EnvironmentException(f'{lang!r} compiler binary not defined in cross or native file')
        compilers = [[x] for x in defaults[lang]]
        ccache = BinaryTable.detect_compiler_cache()
    if env.machines.matches_build_machine(for_machine):
        exe_wrap: T.Optional[ExternalProgram] = None
    else:
        exe_wrap = env.get_exe_wrapper()
    return (compilers, ccache, exe_wrap)
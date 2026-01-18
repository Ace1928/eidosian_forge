from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
@functools.lru_cache()
def _get_library_dirs(self, env: 'Environment', elf_class: T.Optional[int]=None) -> 'ImmutableListProtocol[str]':
    dirs = self.get_compiler_dirs(env, 'libraries')
    if elf_class is None or elf_class == 0:
        return dirs
    retval: T.List[str] = []
    for d in dirs:
        files = [f for f in os.listdir(d) if f.endswith('.so') and os.path.isfile(os.path.join(d, f))]
        if not files:
            retval.append(d)
            continue
        for f in files:
            file_to_check = os.path.join(d, f)
            try:
                with open(file_to_check, 'rb') as fd:
                    header = fd.read(5)
                    if header[1:4] != b'ELF' or int(header[4]) == elf_class:
                        retval.append(d)
                break
            except OSError:
                pass
    return retval
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
def _scan_fortran_file_deps(src: Path, srcdir: Path, dirname: Path, tdeps, compiler) -> T.List[str]:
    """
    scan a Fortran file for dependencies. Needs to be distinct from target
    to allow for recursion induced by `include` statements.er

    It makes a number of assumptions, including

    * `use`, `module`, `submodule` name is not on a continuation line

    Regex
    -----

    * `incre` works for `#include "foo.f90"` and `include "foo.f90"`
    * `usere` works for legacy and Fortran 2003 `use` statements
    * `submodre` is for Fortran >= 2008 `submodule`
    """
    incre = re.compile(FORTRAN_INCLUDE_PAT, re.IGNORECASE)
    usere = re.compile(FORTRAN_USE_PAT, re.IGNORECASE)
    submodre = re.compile(FORTRAN_SUBMOD_PAT, re.IGNORECASE)
    mod_files = []
    src = Path(src)
    with src.open(encoding='ascii', errors='ignore') as f:
        for line in f:
            incmatch = incre.match(line)
            if incmatch is not None:
                incfile = src.parent / incmatch.group(1)
                if incfile.suffix.lower()[1:] in compiler.file_suffixes:
                    mod_files.extend(_scan_fortran_file_deps(incfile, srcdir, dirname, tdeps, compiler))
            usematch = usere.match(line)
            if usematch is not None:
                usename = usematch.group(1).lower()
                if usename == 'intrinsic':
                    continue
                if usename not in tdeps:
                    continue
                srcfile = srcdir / tdeps[usename].fname
                if not srcfile.is_file():
                    if srcfile.name != src.name:
                        pass
                    else:
                        continue
                elif srcfile.samefile(src):
                    continue
                mod_name = compiler.module_name_to_filename(usename)
                mod_files.append(str(dirname / mod_name))
            else:
                submodmatch = submodre.match(line)
                if submodmatch is not None:
                    parents = submodmatch.group(1).lower().split(':')
                    assert len(parents) in {1, 2}, f'submodule ancestry must be specified as ancestor:parent but Meson found {parents}'
                    ancestor_child = '_'.join(parents)
                    if ancestor_child not in tdeps:
                        raise MesonException('submodule {} relies on ancestor module {} that was not found.'.format(submodmatch.group(2).lower(), ancestor_child.split('_', maxsplit=1)[0]))
                    submodsrcfile = srcdir / tdeps[ancestor_child].fname
                    if not submodsrcfile.is_file():
                        if submodsrcfile.name != src.name:
                            pass
                        else:
                            continue
                    elif submodsrcfile.samefile(src):
                        continue
                    mod_name = compiler.module_name_to_filename(ancestor_child)
                    mod_files.append(str(dirname / mod_name))
    return mod_files
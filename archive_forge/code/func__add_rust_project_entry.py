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
def _add_rust_project_entry(self, name: str, main_rust_file: str, args: CompilerArgs, from_subproject: bool, proc_macro_dylib_path: T.Optional[str], deps: T.List[RustDep]) -> None:
    raw_edition: T.Optional[str] = mesonlib.first(reversed(args), lambda x: x.startswith('--edition'))
    edition: RUST_EDITIONS = '2015' if not raw_edition else raw_edition.split('=')[-1]
    cfg: T.List[str] = []
    arg_itr: T.Iterator[str] = iter(args)
    for arg in arg_itr:
        if arg == '--cfg':
            cfg.append(next(arg_itr))
        elif arg.startswith('--cfg'):
            cfg.append(arg[len('--cfg'):])
    crate = RustCrate(len(self.rust_crates), name, main_rust_file, edition, deps, cfg, is_workspace_member=not from_subproject, is_proc_macro=proc_macro_dylib_path is not None, proc_macro_dylib_path=proc_macro_dylib_path)
    self.rust_crates[name] = crate
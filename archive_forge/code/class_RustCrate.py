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
@dataclass
class RustCrate:
    order: int
    display_name: str
    root_module: str
    edition: RUST_EDITIONS
    deps: T.List[RustDep]
    cfg: T.List[str]
    is_proc_macro: bool
    is_workspace_member: bool
    proc_macro_dylib_path: T.Optional[str] = None

    def to_json(self) -> T.Dict[str, object]:
        ret: T.Dict[str, object] = {'display_name': self.display_name, 'root_module': self.root_module, 'edition': self.edition, 'cfg': self.cfg, 'is_proc_macro': self.is_proc_macro, 'deps': [d.to_json() for d in self.deps]}
        if self.is_proc_macro:
            assert self.proc_macro_dylib_path is not None, "This shouldn't happen"
            ret['proc_macro_dylib_path'] = self.proc_macro_dylib_path
        return ret
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
def add_dependency_scanner_entries_to_element(self, target: build.BuildTarget, compiler, element, src):
    if not self.should_use_dyndeps_for_target(target):
        return
    if isinstance(target, build.CompileTarget):
        return
    extension = os.path.splitext(src.fname)[1][1:]
    if extension != 'C':
        extension = extension.lower()
    if not (extension in compilers.lang_suffixes['fortran'] or extension in compilers.lang_suffixes['cpp']):
        return
    dep_scan_file = self.get_dep_scan_file_for(target)
    element.add_item('dyndep', dep_scan_file)
    element.add_orderdep(dep_scan_file)
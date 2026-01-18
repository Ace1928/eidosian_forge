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
def _generate_single_compile_base_args(self, target: build.BuildTarget, compiler: 'Compiler') -> 'CompilerArgs':
    base_proxy = target.get_options()
    commands = compiler.compiler_args()
    commands += compiler.gnu_symbol_visibility_args(target.gnu_symbol_visibility)
    commands += compilers.get_base_compile_args(base_proxy, compiler)
    return commands
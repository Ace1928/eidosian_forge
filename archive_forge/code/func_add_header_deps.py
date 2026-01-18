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
def add_header_deps(self, target, ninja_element, header_deps):
    for d in header_deps:
        if isinstance(d, File):
            d = d.rel_to_builddir(self.build_to_src)
        elif not self.has_dir_part(d):
            d = os.path.join(self.get_target_private_dir(target), d)
        ninja_element.add_dep(d)
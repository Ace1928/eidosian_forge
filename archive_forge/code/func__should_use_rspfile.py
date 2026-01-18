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
def _should_use_rspfile(self):
    if self.rulename == 'phony':
        return False
    if not self.rule.rspable:
        return False
    infilenames = ' '.join([ninja_quote(i, True) for i in self.infilenames])
    outfilenames = ' '.join([ninja_quote(i, True) for i in self.outfilenames])
    return self.rule.length_estimate(infilenames, outfilenames, self.elems) >= rsp_threshold
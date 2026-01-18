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
def add_build(self, build: NinjaBuildElement) -> None:
    build.check_outputs()
    self.build_elements.append(build)
    if build.rulename != 'phony':
        if build.rulename in self.ruledict:
            build.rule = self.ruledict[build.rulename]
        else:
            mlog.warning(f'build statement for {build.outfilenames} references nonexistent rule {build.rulename}')
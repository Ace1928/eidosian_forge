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
def __generate_sources_structure(self, root: Path, structured_sources: build.StructuredSources) -> T.Tuple[T.List[str], T.Optional[str]]:
    first_file: T.Optional[str] = None
    orderdeps: T.List[str] = []
    for path, files in structured_sources.sources.items():
        for file in files:
            if isinstance(file, File):
                out = root / path / Path(file.fname).name
                orderdeps.append(str(out))
                self._generate_copy_target(file, out)
                if first_file is None:
                    first_file = str(out)
            else:
                for f in file.get_outputs():
                    out = root / path / f
                    orderdeps.append(str(out))
                    self._generate_copy_target(str(Path(file.subdir) / f), out)
                    if first_file is None:
                        first_file = str(out)
    return (orderdeps, first_file)
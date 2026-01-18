from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
class CompileCheckMode(enum.Enum):
    PREPROCESS = 'preprocess'
    COMPILE = 'compile'
    LINK = 'link'
from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
class CMakeSkipCompilerTest(Enum):
    ALWAYS = 'always'
    NEVER = 'never'
    DEP_ONLY = 'dep_only'
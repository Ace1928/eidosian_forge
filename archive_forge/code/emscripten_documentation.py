from __future__ import annotations
import os.path
import typing as T
from ... import coredata
from ... import mesonlib
from ...mesonlib import OptionKey
from ...mesonlib import LibType
from mesonbuild.compilers.compilers import CompileCheckMode
Provides a mixin for shared code between C and C++ Emscripten compilers.
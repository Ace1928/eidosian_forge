from __future__ import annotations
import re
import typing as T
from ..mesonlib import listify, version_compare
from ..compilers.cuda import CudaCompiler
from ..interpreter.type_checking import NoneType
from . import NewExtensionModule, ModuleInfo
from ..interpreterbase import (
class ArchFlagsKwargs(TypedDict):
    detected: T.Optional[T.List[str]]
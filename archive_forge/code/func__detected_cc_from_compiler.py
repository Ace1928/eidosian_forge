from __future__ import annotations
import re
import typing as T
from ..mesonlib import listify, version_compare
from ..compilers.cuda import CudaCompiler
from ..interpreter.type_checking import NoneType
from . import NewExtensionModule, ModuleInfo
from ..interpreterbase import (
@staticmethod
def _detected_cc_from_compiler(c: T.Union[str, CudaCompiler]) -> T.List[str]:
    if isinstance(c, CudaCompiler):
        return [c.detected_cc]
    return []
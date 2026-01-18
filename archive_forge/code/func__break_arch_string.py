from __future__ import annotations
import re
import typing as T
from ..mesonlib import listify, version_compare
from ..compilers.cuda import CudaCompiler
from ..interpreter.type_checking import NoneType
from . import NewExtensionModule, ModuleInfo
from ..interpreterbase import (
@staticmethod
def _break_arch_string(s: str) -> T.List[str]:
    s = re.sub('[ \t\r\n,;]+', ';', s)
    return s.strip(';').split(';')
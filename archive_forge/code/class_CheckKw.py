from __future__ import annotations
import typing as T
from .. import mesonlib, mlog
from .. import build
from ..compilers import Compiler
from ..interpreter.type_checking import BT_SOURCES_KW, STATIC_LIB_KWS
from ..interpreterbase.decorators import KwargInfo, permittedKwargs, typed_pos_args, typed_kwargs
from . import ExtensionModule, ModuleInfo
class CheckKw(kwtypes.StaticLibrary):
    compiler: Compiler
    mmx: SourcesVarargsType
    sse: SourcesVarargsType
    sse2: SourcesVarargsType
    sse3: SourcesVarargsType
    ssse3: SourcesVarargsType
    sse41: SourcesVarargsType
    sse42: SourcesVarargsType
    avx: SourcesVarargsType
    avx2: SourcesVarargsType
    neon: SourcesVarargsType
from __future__ import annotations
import copy
import functools
import os.path
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import MesonException, version_compare, OptionKey
from .compilers import (
from .c_function_attributes import CXX_FUNC_ATTRIBUTES, C_FUNC_ATTRIBUTES
from .mixins.clike import CLikeCompiler
from .mixins.ccrx import CcrxCompiler
from .mixins.ti import TICompiler
from .mixins.arm import ArmCompiler, ArmclangCompiler
from .mixins.visualstudio import MSVCCompiler, ClangClCompiler
from .mixins.gnu import GnuCompiler, gnu_common_warning_args, gnu_cpp_warning_args
from .mixins.intel import IntelGnuLikeCompiler, IntelVisualStudioLikeCompiler
from .mixins.clang import ClangCompiler
from .mixins.elbrus import ElbrusCompiler
from .mixins.pgi import PGICompiler
from .mixins.emscripten import EmscriptenMixin
from .mixins.metrowerks import MetrowerksCompiler
from .mixins.metrowerks import mwccarm_instruction_set_args, mwcceppc_instruction_set_args
class MetrowerksCPPCompilerEmbeddedPowerPC(MetrowerksCompiler, CPPCompiler):
    id = 'mwcceppc'

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        CPPCompiler.__init__(self, ccache, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        MetrowerksCompiler.__init__(self)

    def get_instruction_set_args(self, instruction_set: str) -> T.Optional[T.List[str]]:
        return mwcceppc_instruction_set_args.get(instruction_set, None)

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = CPPCompiler.get_options(self)
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        opts[key].choices = ['none']
        return opts

    def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        args: T.List[str] = []
        std = options[OptionKey('std', machine=self.for_machine, lang=self.language)]
        if std.value != 'none':
            args.append('-lang ' + std.value)
        return args
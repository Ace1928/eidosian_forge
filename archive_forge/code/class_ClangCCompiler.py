from __future__ import annotations
import os.path
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import MesonException, version_compare, OptionKey
from .c_function_attributes import C_FUNC_ATTRIBUTES
from .mixins.clike import CLikeCompiler
from .mixins.ccrx import CcrxCompiler
from .mixins.xc16 import Xc16Compiler
from .mixins.compcert import CompCertCompiler
from .mixins.ti import TICompiler
from .mixins.arm import ArmCompiler, ArmclangCompiler
from .mixins.visualstudio import MSVCCompiler, ClangClCompiler
from .mixins.gnu import GnuCompiler
from .mixins.gnu import gnu_common_warning_args, gnu_c_warning_args
from .mixins.intel import IntelGnuLikeCompiler, IntelVisualStudioLikeCompiler
from .mixins.clang import ClangCompiler
from .mixins.elbrus import ElbrusCompiler
from .mixins.pgi import PGICompiler
from .mixins.emscripten import EmscriptenMixin
from .mixins.metrowerks import MetrowerksCompiler
from .mixins.metrowerks import mwccarm_instruction_set_args, mwcceppc_instruction_set_args
from .compilers import (
class ClangCCompiler(_ClangCStds, ClangCompiler, CCompiler):

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, defines: T.Optional[T.Dict[str, str]]=None, full_version: T.Optional[str]=None):
        CCompiler.__init__(self, ccache, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        ClangCompiler.__init__(self, defines)
        default_warn_args = ['-Wall', '-Winvalid-pch']
        self.warn_args = {'0': [], '1': default_warn_args, '2': default_warn_args + ['-Wextra'], '3': default_warn_args + ['-Wextra', '-Wpedantic'], 'everything': ['-Weverything']}

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = super().get_options()
        if self.info.is_windows() or self.info.is_cygwin():
            opts.update({OptionKey('winlibs', machine=self.for_machine, lang=self.language): coredata.UserArrayOption('Standard Win libraries to link against', gnu_winlibs)})
        return opts

    def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        args = []
        std = options[OptionKey('std', machine=self.for_machine, lang=self.language)]
        if std.value != 'none':
            args.append('-std=' + std.value)
        return args

    def get_option_link_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        if self.info.is_windows() or self.info.is_cygwin():
            libs = options[OptionKey('winlibs', machine=self.for_machine, lang=self.language)].value.copy()
            assert isinstance(libs, list)
            for l in libs:
                assert isinstance(l, str)
            return libs
        return []
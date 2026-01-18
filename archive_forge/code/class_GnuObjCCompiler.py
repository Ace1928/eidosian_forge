from __future__ import annotations
import typing as T
from .. import coredata
from ..mesonlib import OptionKey
from .compilers import Compiler
from .mixins.clike import CLikeCompiler
from .mixins.gnu import GnuCompiler, gnu_common_warning_args, gnu_objc_warning_args
from .mixins.clang import ClangCompiler
class GnuObjCCompiler(GnuCompiler, ObjCCompiler):

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, defines: T.Optional[T.Dict[str, str]]=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        ObjCCompiler.__init__(self, ccache, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        GnuCompiler.__init__(self, defines)
        default_warn_args = ['-Wall', '-Winvalid-pch']
        self.warn_args = {'0': [], '1': default_warn_args, '2': default_warn_args + ['-Wextra'], '3': default_warn_args + ['-Wextra', '-Wpedantic'], 'everything': default_warn_args + ['-Wextra', '-Wpedantic'] + self.supported_warn_args(gnu_common_warning_args) + self.supported_warn_args(gnu_objc_warning_args)}
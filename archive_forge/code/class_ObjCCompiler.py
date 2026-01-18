from __future__ import annotations
import typing as T
from .. import coredata
from ..mesonlib import OptionKey
from .compilers import Compiler
from .mixins.clike import CLikeCompiler
from .mixins.gnu import GnuCompiler, gnu_common_warning_args, gnu_objc_warning_args
from .mixins.clang import ClangCompiler
class ObjCCompiler(CLikeCompiler, Compiler):
    language = 'objc'

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrap: T.Optional['ExternalProgram'], linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        Compiler.__init__(self, ccache, exelist, version, for_machine, info, is_cross=is_cross, full_version=full_version, linker=linker)
        CLikeCompiler.__init__(self, exe_wrap)

    @staticmethod
    def get_display_language() -> str:
        return 'Objective-C'

    def sanity_check(self, work_dir: str, environment: 'Environment') -> None:
        code = '#import<stddef.h>\nint main(void) { return 0; }\n'
        return self._sanity_check_impl(work_dir, environment, 'sanitycheckobjc.m', code)
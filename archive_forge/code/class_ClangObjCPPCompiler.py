from __future__ import annotations
import typing as T
from .. import coredata
from ..mesonlib import OptionKey
from .mixins.clike import CLikeCompiler
from .compilers import Compiler
from .mixins.gnu import GnuCompiler, gnu_common_warning_args, gnu_objc_warning_args
from .mixins.clang import ClangCompiler
class ClangObjCPPCompiler(ClangCompiler, ObjCPPCompiler):

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, defines: T.Optional[T.Dict[str, str]]=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        ObjCPPCompiler.__init__(self, ccache, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        ClangCompiler.__init__(self, defines)
        default_warn_args = ['-Wall', '-Winvalid-pch']
        self.warn_args = {'0': [], '1': default_warn_args, '2': default_warn_args + ['-Wextra'], '3': default_warn_args + ['-Wextra', '-Wpedantic'], 'everything': ['-Weverything']}

    def get_options(self) -> 'coredata.MutableKeyedOptionDictType':
        opts = super().get_options()
        opts.update({OptionKey('std', machine=self.for_machine, lang='cpp'): coredata.UserComboOption('C++ language standard to use', ['none', 'c++98', 'c++11', 'c++14', 'c++17', 'c++20', 'c++2b', 'gnu++98', 'gnu++11', 'gnu++14', 'gnu++17', 'gnu++20', 'gnu++2b'], 'none')})
        return opts

    def get_option_compile_args(self, options: 'coredata.KeyedOptionDictType') -> T.List[str]:
        args = []
        std = options[OptionKey('std', machine=self.for_machine, lang='cpp')]
        if std.value != 'none':
            args.append('-std=' + std.value)
        return args
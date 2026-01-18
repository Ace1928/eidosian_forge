from __future__ import annotations
import typing as T
import os
from .. import coredata
from .compilers import (
from .mixins.clike import CLikeCompiler
from .mixins.gnu import GnuCompiler,  gnu_optimization_args
from .mixins.intel import IntelGnuLikeCompiler, IntelVisualStudioLikeCompiler
from .mixins.clang import ClangCompiler
from .mixins.elbrus import ElbrusCompiler
from .mixins.pgi import PGICompiler
from mesonbuild.mesonlib import (
class IntelClFortranCompiler(IntelVisualStudioLikeCompiler, FortranCompiler):
    file_suffixes = ('f90', 'f', 'for', 'ftn', 'fpp')
    always_args = ['/nologo']

    def __init__(self, exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', target: str, exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        FortranCompiler.__init__(self, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        IntelVisualStudioLikeCompiler.__init__(self, target)
        default_warn_args = ['/warn:general', '/warn:truncated_source']
        self.warn_args = {'0': [], '1': default_warn_args, '2': default_warn_args + ['/warn:unused'], '3': ['/warn:all'], 'everything': ['/warn:all']}

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = FortranCompiler.get_options(self)
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        opts[key].choices = ['none', 'legacy', 'f95', 'f2003', 'f2008', 'f2018']
        return opts

    def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        args: T.List[str] = []
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        std = options[key]
        stds = {'legacy': 'none', 'f95': 'f95', 'f2003': 'f03', 'f2008': 'f08', 'f2018': 'f18'}
        if std.value != 'none':
            args.append('/stand:' + stds[std.value])
        return args

    def get_module_outdir_args(self, path: str) -> T.List[str]:
        return ['/module:' + path]
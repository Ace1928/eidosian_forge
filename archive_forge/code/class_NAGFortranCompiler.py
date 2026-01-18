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
class NAGFortranCompiler(FortranCompiler):
    id = 'nagfor'

    def __init__(self, exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        FortranCompiler.__init__(self, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        self.warn_args = {'0': ['-w=all'], '1': [], '2': [], '3': [], 'everything': []}

    def get_always_args(self) -> T.List[str]:
        return self.get_nagfor_quiet(self.version)

    def get_module_outdir_args(self, path: str) -> T.List[str]:
        return ['-mdir', path]

    @staticmethod
    def get_nagfor_quiet(version: str) -> T.List[str]:
        return ['-quiet'] if version_compare(version, '>=7100') else []

    def get_pic_args(self) -> T.List[str]:
        return ['-PIC']

    def get_preprocess_only_args(self) -> T.List[str]:
        return ['-fpp']

    def get_std_exe_link_args(self) -> T.List[str]:
        return self.get_always_args()

    def openmp_flags(self) -> T.List[str]:
        return ['-openmp']
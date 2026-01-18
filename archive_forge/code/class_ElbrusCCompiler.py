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
class ElbrusCCompiler(ElbrusCompiler, CCompiler):

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, defines: T.Optional[T.Dict[str, str]]=None, full_version: T.Optional[str]=None):
        CCompiler.__init__(self, ccache, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        ElbrusCompiler.__init__(self)

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = CCompiler.get_options(self)
        stds = ['c89', 'c9x', 'c99', 'gnu89', 'gnu9x', 'gnu99']
        stds += ['iso9899:1990', 'iso9899:199409', 'iso9899:1999']
        if version_compare(self.version, '>=1.20.00'):
            stds += ['c11', 'gnu11']
        if version_compare(self.version, '>=1.21.00') and version_compare(self.version, '<1.22.00'):
            stds += ['c90', 'c1x', 'gnu90', 'gnu1x', 'iso9899:2011']
        if version_compare(self.version, '>=1.23.00'):
            stds += ['c90', 'c1x', 'gnu90', 'gnu1x', 'iso9899:2011']
        if version_compare(self.version, '>=1.26.00'):
            stds += ['c17', 'c18', 'iso9899:2017', 'iso9899:2018', 'gnu17', 'gnu18']
        std_opt = opts[OptionKey('std', machine=self.for_machine, lang=self.language)]
        assert isinstance(std_opt, coredata.UserStdOption), 'for mypy'
        std_opt.set_versions(stds)
        return opts

    def has_function(self, funcname: str, prefix: str, env: 'Environment', *, extra_args: T.Optional[T.List[str]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> T.Tuple[bool, bool]:
        if funcname == 'lchmod':
            return (False, False)
        else:
            return super().has_function(funcname, prefix, env, extra_args=extra_args, dependencies=dependencies)
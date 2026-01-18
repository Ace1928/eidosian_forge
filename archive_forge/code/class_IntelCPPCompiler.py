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
class IntelCPPCompiler(IntelGnuLikeCompiler, CPPCompiler):

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        CPPCompiler.__init__(self, ccache, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        IntelGnuLikeCompiler.__init__(self)
        self.lang_header = 'c++-header'
        default_warn_args = ['-Wall', '-w3', '-Wpch-messages']
        self.warn_args = {'0': [], '1': default_warn_args + ['-diag-disable:remark'], '2': default_warn_args + ['-Wextra', '-diag-disable:remark'], '3': default_warn_args + ['-Wextra', '-diag-disable:remark'], 'everything': default_warn_args + ['-Wextra']}

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = CPPCompiler.get_options(self)
        c_stds = ['c++98', 'c++03']
        g_stds = ['gnu++98', 'gnu++03']
        if version_compare(self.version, '>=15.0.0'):
            c_stds += ['c++11', 'c++14']
            g_stds += ['gnu++11']
        if version_compare(self.version, '>=16.0.0'):
            c_stds += ['c++17']
        if version_compare(self.version, '>=17.0.0'):
            g_stds += ['gnu++14']
        if version_compare(self.version, '>=19.1.0'):
            c_stds += ['c++2a']
            g_stds += ['gnu++2a']
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        opts.update({key.evolve('eh'): coredata.UserComboOption('C++ exception handling type.', ['none', 'default', 'a', 's', 'sc'], 'default'), key.evolve('rtti'): coredata.UserBooleanOption('Enable RTTI', True), key.evolve('debugstl'): coredata.UserBooleanOption('STL debug mode', False)})
        std_opt = opts[key]
        assert isinstance(std_opt, coredata.UserStdOption), 'for mypy'
        std_opt.set_versions(c_stds + g_stds)
        return opts

    def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        args: T.List[str] = []
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        std = options[key]
        if std.value != 'none':
            remap_cpp03 = {'c++03': 'c++98', 'gnu++03': 'gnu++98'}
            args.append('-std=' + remap_cpp03.get(std.value, std.value))
        if options[key.evolve('eh')].value == 'none':
            args.append('-fno-exceptions')
        if not options[key.evolve('rtti')].value:
            args.append('-fno-rtti')
        if options[key.evolve('debugstl')].value:
            args.append('-D_GLIBCXX_DEBUG=1')
        return args

    def get_option_link_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        return []
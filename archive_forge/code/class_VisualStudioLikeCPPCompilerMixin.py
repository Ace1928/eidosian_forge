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
class VisualStudioLikeCPPCompilerMixin(CompilerMixinBase):
    """Mixin for C++ specific method overrides in MSVC-like compilers."""
    VC_VERSION_MAP = {'none': (True, None), 'vc++11': (True, 11), 'vc++14': (True, 14), 'vc++17': (True, 17), 'vc++20': (True, 20), 'vc++latest': (True, 'latest'), 'c++11': (False, 11), 'c++14': (False, 14), 'c++17': (False, 17), 'c++20': (False, 20), 'c++latest': (False, 'latest')}

    def get_option_link_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        key = OptionKey('winlibs', machine=self.for_machine, lang=self.language)
        return T.cast('T.List[str]', options[key].value[:])

    def _get_options_impl(self, opts: 'MutableKeyedOptionDictType', cpp_stds: T.List[str]) -> 'MutableKeyedOptionDictType':
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        opts.update({key.evolve('eh'): coredata.UserComboOption('C++ exception handling type.', ['none', 'default', 'a', 's', 'sc'], 'default'), key.evolve('rtti'): coredata.UserBooleanOption('Enable RTTI', True), key.evolve('winlibs'): coredata.UserArrayOption('Windows libs to link against.', msvc_winlibs)})
        std_opt = opts[key]
        assert isinstance(std_opt, coredata.UserStdOption), 'for mypy'
        std_opt.set_versions(cpp_stds)
        return opts

    def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        args: T.List[str] = []
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        eh = options[key.evolve('eh')]
        if eh.value == 'default':
            args.append('/EHsc')
        elif eh.value == 'none':
            args.append('/EHs-c-')
        else:
            args.append('/EH' + eh.value)
        if not options[key.evolve('rtti')].value:
            args.append('/GR-')
        permissive, ver = self.VC_VERSION_MAP[options[key].value]
        if ver is not None:
            args.append(f'/std:c++{ver}')
        if not permissive:
            args.append('/permissive-')
        return args

    def get_compiler_check_args(self, mode: CompileCheckMode) -> T.List[str]:
        return Compiler.get_compiler_check_args(self, mode)
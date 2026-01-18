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
class _StdCPPLibMixin(CompilerMixinBase):
    """Detect whether to use libc++ or libstdc++."""

    @functools.lru_cache(None)
    def language_stdlib_only_link_flags(self, env: Environment) -> T.List[str]:
        """Detect the C++ stdlib and default search dirs

        As an optimization, this method will cache the value, to avoid building the same values over and over

        :param env: An Environment object
        :raises MesonException: If a stdlib cannot be determined
        """
        search_dirs = [f'-L{d}' for d in self.get_compiler_dirs(env, 'libraries')]
        machine = env.machines[self.for_machine]
        assert machine is not None, 'for mypy'
        search_order: T.List[str] = []
        if machine.system in {'android', 'darwin', 'dragonfly', 'freebsd', 'netbsd', 'openbsd'}:
            search_order = ['c++', 'stdc++']
        else:
            search_order = ['stdc++', 'c++']
        for lib in search_order:
            if self.find_library(lib, env, []) is not None:
                return search_dirs + [f'-l{lib}']
        raise MesonException('Could not detect either libc++ or libstdc++ as your C++ stdlib implementation.')
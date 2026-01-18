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
def _test_cpp_std_arg(self, cpp_std_value: str) -> bool:
    assert cpp_std_value.startswith('-std=')
    CPP_TEST = 'int i = static_cast<int>(0);'
    with self.compile(CPP_TEST, extra_args=[cpp_std_value], mode=CompileCheckMode.COMPILE) as p:
        if p.returncode == 0:
            mlog.debug(f'Compiler accepts {cpp_std_value}:', 'YES')
            return True
        else:
            mlog.debug(f'Compiler accepts {cpp_std_value}:', 'NO')
            return False
from __future__ import annotations
import typing as T
from .. import coredata
from ..mesonlib import OptionKey
from .compilers import Compiler
from .mixins.clike import CLikeCompiler
from .mixins.gnu import GnuCompiler, gnu_common_warning_args, gnu_objc_warning_args
from .mixins.clang import ClangCompiler
class AppleClangObjCCompiler(ClangObjCCompiler):
    """Handle the differences between Apple's clang and vanilla clang."""
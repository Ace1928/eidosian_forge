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
class ArmLtdFlangFortranCompiler(FlangFortranCompiler):
    id = 'armltdflang'
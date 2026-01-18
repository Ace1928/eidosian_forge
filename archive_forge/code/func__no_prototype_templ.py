from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
@staticmethod
def _no_prototype_templ() -> T.Tuple[str, str]:
    """
        Try to find the function without a prototype from a header by defining
        our own dummy prototype and trying to link with the C library (and
        whatever else the compiler links in by default). This is very similar
        to the check performed by Autoconf for AC_CHECK_FUNCS.
        """
    head = '\n        #define {func} meson_disable_define_of_{func}\n        {prefix}\n        #include <limits.h>\n        #undef {func}\n        '
    head += '\n        #ifdef __cplusplus\n        extern "C"\n        #endif\n        char {func} (void);\n        '
    main = '\n        int main(void) {{\n          return {func} ();\n        }}'
    return (head, main)
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
def _have_prototype_templ() -> T.Tuple[str, str]:
    """
        Returns a head-er and main() call that uses the headers listed by the
        user for the function prototype while checking if a function exists.
        """
    head = '{prefix}\n#include <limits.h>\n'
    main = '\nint main(void) {{\n            void *a = (void*) &{func};\n            long long b = (long long) a;\n            return (int) b;\n        }}'
    return (head, main)
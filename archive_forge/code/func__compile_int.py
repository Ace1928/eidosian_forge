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
def _compile_int(self, expression: str, prefix: str, env: 'Environment', extra_args: T.Union[None, T.List[str], T.Callable[[CompileCheckMode], T.List[str]]], dependencies: T.Optional[T.List['Dependency']]) -> bool:
    t = f'{prefix}\n        #include <stddef.h>\n        int main(void) {{ static int a[1-2*!({expression})]; a[0]=0; return 0; }}'
    return self.compiles(t, env, extra_args=extra_args, dependencies=dependencies)[0]
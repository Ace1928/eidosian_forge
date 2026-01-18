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
def _find_framework_real(self, name: str, env: 'Environment', extra_dirs: T.List[str], allow_system: bool) -> T.Optional[T.List[str]]:
    code = 'int main(void) { return 0; }'
    link_args: T.List[str] = []
    for d in extra_dirs:
        link_args += ['-F' + d]
    extra_args = [] if allow_system else ['-Z', '-L/usr/lib']
    link_args += ['-framework', name]
    if self.links(code, env, extra_args=extra_args + link_args, disable_cache=True)[0]:
        return link_args
    return None
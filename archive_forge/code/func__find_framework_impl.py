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
def _find_framework_impl(self, name: str, env: 'Environment', extra_dirs: T.List[str], allow_system: bool) -> T.Optional[T.List[str]]:
    if isinstance(extra_dirs, str):
        extra_dirs = [extra_dirs]
    key = (tuple(self.exelist), name, tuple(extra_dirs), allow_system)
    if key in self.find_framework_cache:
        value = self.find_framework_cache[key]
    else:
        value = self._find_framework_real(name, env, extra_dirs, allow_system)
        self.find_framework_cache[key] = value
    if value is None:
        return None
    return value.copy()
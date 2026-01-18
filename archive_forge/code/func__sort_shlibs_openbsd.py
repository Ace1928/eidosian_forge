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
def _sort_shlibs_openbsd(libs: T.List[str]) -> T.List[str]:

    def tuple_key(x: str) -> T.Tuple[int, ...]:
        ver = x.rsplit('.so.', maxsplit=1)[1]
        return tuple((int(i) for i in ver.split('.')))
    filtered: T.List[str] = []
    for lib in libs:
        ret = lib.rsplit('.so.', maxsplit=1)
        if len(ret) != 2:
            continue
        try:
            tuple((int(i) for i in ret[1].split('.')))
        except ValueError:
            continue
        filtered.append(lib)
    return sorted(filtered, key=tuple_key, reverse=True)
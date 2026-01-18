from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class GnuGoldDynamicLinker(GnuDynamicLinker):
    id = 'ld.gold'

    def get_thinlto_cache_args(self, path: str) -> T.List[str]:
        return ['-Wl,-plugin-opt,cache-dir=' + path]
from __future__ import annotations
import abc
import functools
import os
import multiprocessing
import pathlib
import re
import subprocess
import typing as T
from ... import mesonlib
from ... import mlog
from ...mesonlib import OptionKey
from mesonbuild.compilers.compilers import CompileCheckMode
@functools.lru_cache()
def _get_search_dirs(self, env: 'Environment') -> str:
    extra_args = ['--print-search-dirs']
    with self._build_wrapper('', env, extra_args=extra_args, dependencies=None, mode=CompileCheckMode.COMPILE, want_output=True) as p:
        return p.stdout
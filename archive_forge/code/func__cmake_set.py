from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def _cmake_set(self, tline: CMakeTraceLine) -> None:
    """Handler for the CMake set() function in all varieties.

        comes in three flavors:
        set(<var> <value> [PARENT_SCOPE])
        set(<var> <value> CACHE <type> <docstring> [FORCE])
        set(ENV{<var>} <value>)

        We don't support the ENV variant, and any uses of it will be ignored
        silently. the other two variates are supported, with some caveats:
        - we don't properly handle scoping, so calls to set() inside a
          function without PARENT_SCOPE set could incorrectly shadow the
          outer scope.
        - We don't honor the type of CACHE arguments
        """
    cache_type = None
    cache_force = 'FORCE' in tline.args
    try:
        cache_idx = tline.args.index('CACHE')
        cache_type = tline.args[cache_idx + 1]
    except (ValueError, IndexError):
        pass
    args = []
    for i in tline.args:
        if not i or i == 'PARENT_SCOPE':
            continue
        if i == 'CACHE':
            break
        args.append(i)
    if len(args) < 1:
        return self._gen_exception('set', 'requires at least one argument', tline)
    identifier = args.pop(0)
    value = ' '.join(args)
    if cache_type:
        if identifier not in self.cache or cache_force:
            self.cache[identifier] = CMakeCacheEntry(value.split(';'), cache_type)
    if not value:
        if identifier in self.vars:
            del self.vars[identifier]
    else:
        self.vars[identifier] = value.split(';')
        self.vars_by_file.setdefault(tline.file, {})[identifier] = value.split(';')
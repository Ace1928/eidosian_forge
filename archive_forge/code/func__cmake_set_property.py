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
def _cmake_set_property(self, tline: CMakeTraceLine) -> None:
    args = list(tline.args)
    scope = args.pop(0)
    append = False
    targets = []
    while args:
        curr = args.pop(0)
        if curr in {'APPEND', 'APPEND_STRING'}:
            append = True
            continue
        if curr == 'PROPERTY':
            break
        targets += curr.split(';')
    if not args:
        return self._gen_exception('set_property', 'failed to parse argument list', tline)
    if len(args) == 1:
        return
    identifier = args.pop(0)
    if self.trace_format == 'human':
        value = ' '.join(args).split(';')
    else:
        value = [y for x in args for y in x.split(';')]
    if not value:
        return

    def do_target(t: str) -> None:
        if t not in self.targets:
            return self._gen_exception('set_property', f'TARGET {t} not found', tline)
        tgt = self.targets[t]
        if identifier not in tgt.properties:
            tgt.properties[identifier] = []
        if append:
            tgt.properties[identifier] += value
        else:
            tgt.properties[identifier] = value

    def do_source(src: str) -> None:
        if identifier != 'HEADER_FILE_ONLY' or not self._str_to_bool(value):
            return
        current_src_dir = self.var_to_str('MESON_PS_CMAKE_CURRENT_SOURCE_DIR')
        if not current_src_dir:
            mlog.warning(textwrap.dedent('                    CMake trace: set_property(SOURCE) called before the preload script was loaded.\n                    Unable to determine CMAKE_CURRENT_SOURCE_DIR. This can lead to build errors.\n                '))
            current_src_dir = '.'
        cur_p = Path(current_src_dir)
        src_p = Path(src)
        if not src_p.is_absolute():
            src_p = cur_p / src_p
        self.explicit_headers.add(src_p)
    if scope == 'TARGET':
        for i in targets:
            do_target(i)
    elif scope == 'SOURCE':
        files = self._guess_files(targets)
        for i in files:
            do_source(i)
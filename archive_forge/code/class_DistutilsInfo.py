from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
class DistutilsInfo(object):

    def __init__(self, source=None, exn=None):
        self.values = {}
        if source is not None:
            for line in line_iter(source):
                line = line.lstrip()
                if not line:
                    continue
                if line[0] != '#':
                    break
                line = line[1:].lstrip()
                kind = next((k for k in ('distutils:', 'cython:') if line.startswith(k)), None)
                if kind is not None:
                    key, _, value = [s.strip() for s in line[len(kind):].partition('=')]
                    type = distutils_settings.get(key, None)
                    if line.startswith('cython:') and type is None:
                        continue
                    if type in (list, transitive_list):
                        value = parse_list(value)
                        if key == 'define_macros':
                            value = [tuple(macro.split('=', 1)) if '=' in macro else (macro, None) for macro in value]
                    if type is bool_or:
                        value = _legacy_strtobool(value)
                    self.values[key] = value
        elif exn is not None:
            for key in distutils_settings:
                if key in ('name', 'sources', 'np_pythran'):
                    continue
                value = getattr(exn, key, None)
                if value:
                    self.values[key] = value

    def merge(self, other):
        if other is None:
            return self
        for key, value in other.values.items():
            type = distutils_settings[key]
            if type is transitive_str and key not in self.values:
                self.values[key] = value
            elif type is transitive_list:
                if key in self.values:
                    all = self.values[key][:]
                    for v in value:
                        if v not in all:
                            all.append(v)
                    value = all
                self.values[key] = value
            elif type is bool_or:
                self.values[key] = self.values.get(key, False) | value
        return self

    def subs(self, aliases):
        if aliases is None:
            return self
        resolved = DistutilsInfo()
        for key, value in self.values.items():
            type = distutils_settings[key]
            if type in [list, transitive_list]:
                new_value_list = []
                for v in value:
                    if v in aliases:
                        v = aliases[v]
                    if isinstance(v, list):
                        new_value_list += v
                    else:
                        new_value_list.append(v)
                value = new_value_list
            elif value in aliases:
                value = aliases[value]
            resolved.values[key] = value
        return resolved

    def apply(self, extension):
        for key, value in self.values.items():
            type = distutils_settings[key]
            if type in [list, transitive_list]:
                value = getattr(extension, key) + list(value)
            setattr(extension, key, value)
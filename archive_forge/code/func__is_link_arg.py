from __future__ import annotations
import functools
import typing as T
import os
import re
from ..environment import detect_cpu_family
from .base import DependencyMethods, detect_compiler, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import factory_methods
from .pkgconfig import PkgConfigDependency
def _is_link_arg(self, f: str) -> bool:
    if self.clib_compiler.id == 'intel-cl':
        return f == '/link' or f.startswith('/LIBPATH') or f.endswith('.lib')
    else:
        return f.startswith(('-L', '-l', '-Xlinker')) or f == '-pthread' or (f.startswith('-W') and f != '-Wall' and (not f.startswith('-Werror')))
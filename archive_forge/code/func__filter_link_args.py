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
def _filter_link_args(self, args: T.List[str]) -> T.List[str]:
    """
        MPI wrappers return a bunch of garbage args.
        Drop -O2 and everything that is not needed.
        """
    result = []
    include_next = False
    for f in args:
        if self._is_link_arg(f):
            result.append(f)
            if f in {'-L', '-Xlinker'}:
                include_next = True
        elif include_next:
            include_next = False
            result.append(f)
    return result
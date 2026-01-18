from __future__ import annotations
import abc
import re
import os
import typing as T
from .base import DependencyException, DependencyMethods
from .configtool import ConfigToolDependency
from .detect import packages
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from .factory import DependencyFactory
from .. import mlog
from .. import mesonlib
def _link_with_qt_winmain(self, is_debug: bool, libdir: T.Union[str, T.List[str]]) -> bool:
    libdir = mesonlib.listify(libdir)
    base_name = self.get_qt_winmain_base_name(is_debug)
    qt_winmain = self.clib_compiler.find_library(base_name, self.env, libdir)
    if qt_winmain:
        self.link_args.append(qt_winmain[0])
        return True
    return False
from __future__ import annotations
import glob
import re
import os
import typing as T
from pathlib import Path
from .. import mesonlib
from .. import mlog
from ..environment import detect_cpu_family
from .base import DependencyException, SystemDependency
from .detect import packages
def _find_requested_libraries(self) -> bool:
    all_found = True
    for module in self.requested_modules:
        args = self.clib_compiler.find_library(module, self.env, [self.libdir] if self.libdir else [])
        if args is None:
            self._report_dependency_error(f"Couldn't find requested CUDA module '{module}'")
            all_found = False
        else:
            mlog.debug(f"Link args for CUDA module '{module}' are {args}")
            self.lib_modules[module] = args
    return all_found
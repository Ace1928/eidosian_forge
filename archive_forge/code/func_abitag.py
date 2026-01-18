from __future__ import annotations
import re
import dataclasses
import functools
import typing as T
from pathlib import Path
from .. import mlog
from .. import mesonlib
from .base import DependencyException, SystemDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .misc import threads_factory
@property
def abitag(self) -> str:
    abitag = ''
    abitag += 'S' if self.static else '-'
    abitag += 'M' if self.mt else '-'
    abitag += ' '
    abitag += 's' if self.runtime_static else '-'
    abitag += 'g' if self.runtime_debug else '-'
    abitag += 'y' if self.python_debug else '-'
    abitag += 'd' if self.debug else '-'
    abitag += 'p' if self.stlport else '-'
    abitag += 'n' if self.deprecated_iostreams else '-'
    abitag += ' ' + (self.arch or '???')
    abitag += ' ' + (self.toolset or '?')
    abitag += ' ' + (self.version_lib or 'x_xx')
    return abitag
from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class VisualStudioLinker(VisualStudioLikeLinker, StaticLinker):
    """Microsoft's lib static linker."""
    id = 'lib'

    def __init__(self, exelist: T.List[str], machine: str):
        StaticLinker.__init__(self, exelist)
        VisualStudioLikeLinker.__init__(self, machine)
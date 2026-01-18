from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class GnuDynamicLinker(GnuLikeDynamicLinkerMixin, PosixDynamicLinkerMixin, DynamicLinker):
    """Representation of GNU ld.bfd and ld.gold."""

    def get_accepts_rsp(self) -> bool:
        return True
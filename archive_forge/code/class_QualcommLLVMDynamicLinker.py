from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class QualcommLLVMDynamicLinker(LLVMDynamicLinker):
    """ARM Linker from Snapdragon LLVM ARM Compiler."""
    id = 'ld.qcld'
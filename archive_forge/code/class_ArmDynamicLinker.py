from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class ArmDynamicLinker(PosixDynamicLinkerMixin, DynamicLinker):
    """Linker for the ARM compiler."""
    id = 'armlink'

    def __init__(self, for_machine: mesonlib.MachineChoice, *, version: str='unknown version'):
        super().__init__(['armlink'], for_machine, '', [], version=version)

    def get_accepts_rsp(self) -> bool:
        return False

    def get_std_shared_lib_args(self) -> 'T.NoReturn':
        raise MesonException('The Arm Linkers do not support shared libraries')

    def get_allow_undefined_args(self) -> T.List[str]:
        return []
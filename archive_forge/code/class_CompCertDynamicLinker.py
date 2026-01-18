from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class CompCertDynamicLinker(DynamicLinker):
    """Linker for CompCert C compiler."""
    id = 'ccomp'

    def __init__(self, for_machine: mesonlib.MachineChoice, *, version: str='unknown version'):
        super().__init__(['ccomp'], for_machine, '', [], version=version)

    def get_link_whole_for(self, args: T.List[str]) -> T.List[str]:
        if not args:
            return args
        return self._apply_prefix('-Wl,--whole-archive') + args + self._apply_prefix('-Wl,--no-whole-archive')

    def get_accepts_rsp(self) -> bool:
        return False

    def get_lib_prefix(self) -> str:
        return ''

    def get_std_shared_lib_args(self) -> T.List[str]:
        return []

    def get_output_args(self, outputname: str) -> T.List[str]:
        return [f'-o{outputname}']

    def get_search_args(self, dirname: str) -> T.List[str]:
        return [f'-L{dirname}']

    def get_allow_undefined_args(self) -> T.List[str]:
        return []

    def get_soname_args(self, env: 'Environment', prefix: str, shlib_name: str, suffix: str, soversion: str, darwin_versions: T.Tuple[str, str]) -> T.List[str]:
        raise MesonException(f'{self.id} does not support shared libraries.')

    def build_rpath_args(self, env: 'Environment', build_dir: str, from_dir: str, rpath_paths: T.Tuple[str, ...], build_rpath: str, install_rpath: str) -> T.Tuple[T.List[str], T.Set[bytes]]:
        return ([], set())
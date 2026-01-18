from __future__ import annotations
import os
import typing as T
from ..mesonlib import EnvironmentException, OptionKey, get_meson_command
from .compilers import Compiler
from .mixins.metrowerks import MetrowerksCompiler, mwasmarm_instruction_set_args, mwasmeppc_instruction_set_args
class MetrowerksAsmCompiler(MetrowerksCompiler, Compiler):
    language = 'nasm'

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: 'MachineChoice', info: 'MachineInfo', linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None, is_cross: bool=False):
        Compiler.__init__(self, ccache, exelist, version, for_machine, info, linker, full_version, is_cross)
        MetrowerksCompiler.__init__(self)
        self.warn_args: T.Dict[str, T.List[str]] = {'0': [], '1': [], '2': [], '3': [], 'everything': []}
        self.can_compile_suffixes.add('s')

    def get_crt_compile_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        return []

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return []

    def get_pic_args(self) -> T.List[str]:
        return []

    def needs_static_linker(self) -> bool:
        return True
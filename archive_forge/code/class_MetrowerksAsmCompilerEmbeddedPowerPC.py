from __future__ import annotations
import os
import typing as T
from ..mesonlib import EnvironmentException, OptionKey, get_meson_command
from .compilers import Compiler
from .mixins.metrowerks import MetrowerksCompiler, mwasmarm_instruction_set_args, mwasmeppc_instruction_set_args
class MetrowerksAsmCompilerEmbeddedPowerPC(MetrowerksAsmCompiler):
    id = 'mwasmeppc'

    def get_instruction_set_args(self, instruction_set: str) -> T.Optional[T.List[str]]:
        return mwasmeppc_instruction_set_args.get(instruction_set, None)

    def sanity_check(self, work_dir: str, environment: 'Environment') -> None:
        if self.info.cpu_family not in {'ppc'}:
            raise EnvironmentException(f'ASM compiler {self.id!r} does not support {self.info.cpu_family} CPU family')
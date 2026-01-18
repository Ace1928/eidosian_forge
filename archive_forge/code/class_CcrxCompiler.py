from __future__ import annotations
import os
import typing as T
from ...mesonlib import EnvironmentException
class CcrxCompiler(Compiler):
    if T.TYPE_CHECKING:
        is_cross = True
        can_compile_suffixes: T.Set[str] = set()
    id = 'ccrx'

    def __init__(self) -> None:
        if not self.is_cross:
            raise EnvironmentException('ccrx supports only cross-compilation.')
        self.can_compile_suffixes.add('src')
        default_warn_args: T.List[str] = []
        self.warn_args: T.Dict[str, T.List[str]] = {'0': [], '1': default_warn_args, '2': default_warn_args + [], '3': default_warn_args + [], 'everything': default_warn_args + []}

    def get_pic_args(self) -> T.List[str]:
        return []

    def get_pch_suffix(self) -> str:
        return 'pch'

    def get_pch_use_args(self, pch_dir: str, header: str) -> T.List[str]:
        return []

    def thread_flags(self, env: 'Environment') -> T.List[str]:
        return []

    def get_coverage_args(self) -> T.List[str]:
        return []

    def get_no_stdinc_args(self) -> T.List[str]:
        return []

    def get_no_stdlib_link_args(self) -> T.List[str]:
        return []

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return ccrx_optimization_args[optimization_level]

    def get_debug_args(self, is_debug: bool) -> T.List[str]:
        return ccrx_debug_args[is_debug]

    @classmethod
    def _unix_args_to_native(cls, args: T.List[str], info: MachineInfo) -> T.List[str]:
        result: T.List[str] = []
        for i in args:
            if i.startswith('-D'):
                i = '-define=' + i[2:]
            if i.startswith('-I'):
                i = '-include=' + i[2:]
            if i.startswith('-Wl,-rpath='):
                continue
            elif i == '--print-search-dirs':
                continue
            elif i.startswith('-L'):
                continue
            elif not i.startswith('-lib=') and i.endswith(('.a', '.lib')):
                i = '-lib=' + i
            result.append(i)
        return result

    def compute_parameters_with_absolute_paths(self, parameter_list: T.List[str], build_dir: str) -> T.List[str]:
        for idx, i in enumerate(parameter_list):
            if i[:9] == '-include=':
                parameter_list[idx] = i[:9] + os.path.normpath(os.path.join(build_dir, i[9:]))
        return parameter_list
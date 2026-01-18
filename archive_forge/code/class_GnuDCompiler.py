from __future__ import annotations
import os.path
import re
import subprocess
import typing as T
from .. import mesonlib
from .. import mlog
from ..arglist import CompilerArgs
from ..linkers import RSPFileSyntax
from ..mesonlib import (
from . import compilers
from .compilers import (
from .mixins.gnu import GnuCompiler
from .mixins.gnu import gnu_common_warning_args
class GnuDCompiler(GnuCompiler, DCompiler):
    LINKER_PREFIX = GnuCompiler.LINKER_PREFIX
    id = 'gcc'

    def __init__(self, exelist: T.List[str], version: str, for_machine: MachineChoice, info: 'MachineInfo', arch: str, *, exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None, is_cross: bool=False):
        DCompiler.__init__(self, exelist, version, for_machine, info, arch, exe_wrapper=exe_wrapper, linker=linker, full_version=full_version, is_cross=is_cross)
        GnuCompiler.__init__(self, {})
        default_warn_args = ['-Wall', '-Wdeprecated']
        self.warn_args = {'0': [], '1': default_warn_args, '2': default_warn_args + ['-Wextra'], '3': default_warn_args + ['-Wextra', '-Wpedantic'], 'everything': default_warn_args + ['-Wextra', '-Wpedantic'] + self.supported_warn_args(gnu_common_warning_args)}
        self.base_options = {OptionKey(o) for o in ['b_colorout', 'b_sanitize', 'b_staticpic', 'b_vscrt', 'b_coverage', 'b_pgo', 'b_ndebug']}
        self._has_color_support = version_compare(self.version, '>=4.9')
        self._has_deps_support = version_compare(self.version, '>=7.1')

    def get_colorout_args(self, colortype: str) -> T.List[str]:
        if self._has_color_support:
            super().get_colorout_args(colortype)
        return []

    def get_dependency_gen_args(self, outtarget: str, outfile: str) -> T.List[str]:
        if self._has_deps_support:
            return super().get_dependency_gen_args(outtarget, outfile)
        return []

    def get_warn_args(self, level: str) -> T.List[str]:
        return self.warn_args[level]

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return gdc_optimization_args[optimization_level]

    def compute_parameters_with_absolute_paths(self, parameter_list: T.List[str], build_dir: str) -> T.List[str]:
        for idx, i in enumerate(parameter_list):
            if i[:2] == '-I' or i[:2] == '-L':
                parameter_list[idx] = i[:2] + os.path.normpath(os.path.join(build_dir, i[2:]))
        return parameter_list

    def get_allow_undefined_link_args(self) -> T.List[str]:
        return self.linker.get_allow_undefined_args()

    def get_linker_always_args(self) -> T.List[str]:
        args = super().get_linker_always_args()
        if self.info.is_windows():
            return args
        return args + ['-shared-libphobos']

    def get_assert_args(self, disable: bool) -> T.List[str]:
        if disable:
            return ['-frelease']
        return []
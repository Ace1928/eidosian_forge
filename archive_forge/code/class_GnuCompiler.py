from __future__ import annotations
import abc
import functools
import os
import multiprocessing
import pathlib
import re
import subprocess
import typing as T
from ... import mesonlib
from ... import mlog
from ...mesonlib import OptionKey
from mesonbuild.compilers.compilers import CompileCheckMode
class GnuCompiler(GnuLikeCompiler):
    """
    GnuCompiler represents an actual GCC in its many incarnations.
    Compilers imitating GCC (Clang/Intel) should use the GnuLikeCompiler ABC.
    """
    id = 'gcc'

    def __init__(self, defines: T.Optional[T.Dict[str, str]]):
        super().__init__()
        self.defines = defines or {}
        self.base_options.update({OptionKey('b_colorout'), OptionKey('b_lto_threads')})

    def get_colorout_args(self, colortype: str) -> T.List[str]:
        if mesonlib.version_compare(self.version, '>=4.9.0'):
            return gnu_color_args[colortype][:]
        return []

    def get_warn_args(self, level: str) -> T.List[str]:
        args = super().get_warn_args(level)
        if mesonlib.version_compare(self.version, '<4.8.0') and '-Wpedantic' in args:
            args[args.index('-Wpedantic')] = '-pedantic'
        return args

    def supported_warn_args(self, warn_args_by_version: T.Dict[str, T.List[str]]) -> T.List[str]:
        result: T.List[str] = []
        for version, warn_args in warn_args_by_version.items():
            if mesonlib.version_compare(self.version, '>=' + version):
                result += warn_args
        return result

    def has_builtin_define(self, define: str) -> bool:
        return define in self.defines

    def get_builtin_define(self, define: str) -> T.Optional[str]:
        if define in self.defines:
            return self.defines[define]
        return None

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return gnu_optimization_args[optimization_level]

    def get_pch_suffix(self) -> str:
        return 'gch'

    def openmp_flags(self) -> T.List[str]:
        return ['-fopenmp']

    def has_arguments(self, args: T.List[str], env: 'Environment', code: str, mode: CompileCheckMode) -> T.Tuple[bool, bool]:
        with self._build_wrapper(code, env, args, None, mode) as p:
            result = p.returncode == 0
            if self.language in {'cpp', 'objcpp'} and 'is valid for C/ObjC' in p.stderr:
                result = False
            if self.language in {'c', 'objc'} and 'is valid for C++/ObjC++' in p.stderr:
                result = False
        return (result, p.cached)

    def get_has_func_attribute_extra_args(self, name: str) -> T.List[str]:
        return ['-Werror=attributes']

    def get_prelink_args(self, prelink_name: str, obj_list: T.List[str]) -> T.List[str]:
        return ['-r', '-o', prelink_name] + obj_list

    def get_lto_compile_args(self, *, threads: int=0, mode: str='default') -> T.List[str]:
        if threads == 0:
            if mesonlib.version_compare(self.version, '>= 10.0'):
                return ['-flto=auto']
            return [f'-flto={multiprocessing.cpu_count()}']
        elif threads > 0:
            return [f'-flto={threads}']
        return super().get_lto_compile_args(threads=threads)

    @classmethod
    def use_linker_args(cls, linker: str, version: str) -> T.List[str]:
        if linker == 'mold' and mesonlib.version_compare(version, '>=12.0.1'):
            return ['-fuse-ld=mold']
        return super().use_linker_args(linker, version)

    def get_profile_use_args(self) -> T.List[str]:
        return super().get_profile_use_args() + ['-fprofile-correction']
from __future__ import annotations
import glob
import os
import re
import pathlib
import shutil
import subprocess
import typing as T
import functools
from mesonbuild.interpreterbase.decorators import FeatureDeprecated
from .. import mesonlib, mlog
from ..environment import get_llvm_tool_names
from ..mesonlib import version_compare, version_compare_many, search_version, stringlistify, extract_as_list
from .base import DependencyException, DependencyMethods, detect_compiler, strip_system_includedirs, strip_system_libdirs, SystemDependency, ExternalDependency, DependencyTypeName
from .cmake import CMakeDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .misc import threads_factory
from .pkgconfig import PkgConfigDependency
class LLVMDependencyCMake(CMakeDependency):

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        self.llvm_modules = stringlistify(extract_as_list(kwargs, 'modules'))
        self.llvm_opt_modules = stringlistify(extract_as_list(kwargs, 'optional_modules'))
        compilers = None
        if kwargs.get('native', False):
            compilers = env.coredata.compilers.build
        else:
            compilers = env.coredata.compilers.host
        if not compilers or not all((x in compilers for x in ('c', 'cpp'))):
            ExternalDependency.__init__(self, DependencyTypeName('cmake'), env, kwargs)
            self.found_modules: T.List[str] = []
            self.name = name
            mlog.warning('The LLVM dependency was not found via CMake since both a C and C++ compiler are required.')
            return
        super().__init__(name, env, kwargs, language='cpp', force_use_global_compilers=True)
        if not self.cmakebin.found():
            return
        if not self.is_found:
            return
        if not self.static and version_compare(self.version, '< 7.0') and self.llvm_modules:
            mlog.warning('Before version 7.0 cmake does not export modules for dynamic linking, cannot check required modules')
            return
        inc_dirs = self.traceparser.get_cmake_var('PACKAGE_INCLUDE_DIRS')
        defs = self.traceparser.get_cmake_var('PACKAGE_DEFINITIONS')
        if len(defs) == 1:
            defs = defs[0].split(' ')
        temp = ['-I' + x for x in inc_dirs] + defs
        self.compile_args += [x for x in temp if x not in self.compile_args]
        self.compile_args = strip_system_includedirs(env, self.for_machine, self.compile_args)
        if not self._add_sub_dependency(threads_factory(env, self.for_machine, {})):
            self.is_found = False
            return

    def _main_cmake_file(self) -> str:
        return 'CMakeListsLLVM.txt'

    def llvm_cmake_versions(self) -> T.List[str]:

        def ver_from_suf(req: str) -> str:
            return search_version(req.strip('-') + '.0')

        def version_sorter(a: str, b: str) -> int:
            if version_compare(a, '=' + b):
                return 0
            if version_compare(a, '<' + b):
                return 1
            return -1
        llvm_requested_versions = [ver_from_suf(x) for x in get_llvm_tool_names('') if version_compare(ver_from_suf(x), '>=0')]
        if self.version_reqs:
            llvm_requested_versions = [ver_from_suf(x) for x in get_llvm_tool_names('') if version_compare_many(ver_from_suf(x), self.version_reqs)]
        return sorted(llvm_requested_versions, key=functools.cmp_to_key(version_sorter))

    def _extra_cmake_opts(self) -> T.List[str]:
        return ['-DLLVM_MESON_REQUIRED_MODULES={}'.format(';'.join(self.llvm_modules)), '-DLLVM_MESON_OPTIONAL_MODULES={}'.format(';'.join(self.llvm_opt_modules)), '-DLLVM_MESON_PACKAGE_NAMES={}'.format(';'.join(get_llvm_tool_names(self.name))), '-DLLVM_MESON_VERSIONS={}'.format(';'.join(self.llvm_cmake_versions())), '-DLLVM_MESON_DYLIB={}'.format('OFF' if self.static else 'ON')]

    def _map_module_list(self, modules: T.List[T.Tuple[str, bool]], components: T.List[T.Tuple[str, bool]]) -> T.List[T.Tuple[str, bool]]:
        res = []
        for mod, required in modules:
            cm_targets = self.traceparser.get_cmake_var(f'MESON_LLVM_TARGETS_{mod}')
            if not cm_targets:
                if required:
                    raise self._gen_exception(f'LLVM module {mod} was not found')
                else:
                    mlog.warning('Optional LLVM module', mlog.bold(mod), 'was not found', fatal=False)
                    continue
            for i in cm_targets:
                res += [(i, required)]
        return res

    def _original_module_name(self, module: str) -> str:
        orig_name = self.traceparser.get_cmake_var(f'MESON_TARGET_TO_LLVM_{module}')
        if orig_name:
            return orig_name[0]
        return module
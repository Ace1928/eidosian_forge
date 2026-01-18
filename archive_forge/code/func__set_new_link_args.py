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
def _set_new_link_args(self, environment: 'Environment') -> None:
    """How to set linker args for LLVM versions >= 3.9"""
    try:
        mode = self.get_config_value(['--shared-mode'], 'link_args')[0]
    except IndexError:
        mlog.debug('llvm-config --shared-mode returned an error')
        self.is_found = False
        return
    if not self.static and mode == 'static':
        try:
            self.__check_libfiles(True)
        except DependencyException:
            lib_ext = get_shared_library_suffix(environment, self.for_machine)
            libdir = self.get_config_value(['--libdir'], 'link_args')[0]
            matches = sorted(glob.iglob(os.path.join(libdir, f'libLLVM*{lib_ext}')))
            if not matches:
                if self.required:
                    raise
                self.is_found = False
                return
            self.link_args = self.get_config_value(['--ldflags'], 'link_args')
            libname = os.path.basename(matches[0]).rstrip(lib_ext).lstrip('lib')
            self.link_args.append(f'-l{libname}')
            return
    elif self.static and mode == 'shared':
        try:
            self.__check_libfiles(False)
        except DependencyException:
            if self.required:
                raise
            self.is_found = False
            return
    link_args = ['--link-static', '--system-libs'] if self.static else ['--link-shared']
    self.link_args = self.get_config_value(['--libs', '--ldflags'] + link_args + list(self.required_modules), 'link_args')
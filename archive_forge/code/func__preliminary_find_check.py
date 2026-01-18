from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from ..mesonlib import is_windows, MesonException, PerMachine, stringlistify, extract_as_list
from ..cmake import CMakeExecutor, CMakeTraceParser, CMakeException, CMakeToolchain, CMakeExecScope, check_cmake_args, resolve_cmake_trace_targets, cmake_is_debug
from .. import mlog
import importlib.resources
from pathlib import Path
import functools
import re
import os
import shutil
import textwrap
import typing as T
def _preliminary_find_check(self, name: str, module_path: T.List[str], prefix_path: T.List[str], machine: 'MachineInfo') -> bool:
    lname = str(name).lower()

    def find_module(path: str) -> bool:
        for i in [path, os.path.join(path, 'cmake'), os.path.join(path, 'CMake')]:
            if not self._cached_isdir(i):
                continue
            content = self._cached_listdir(i)
            candidates = ['Find{}.cmake', '{}Config.cmake', '{}-config.cmake']
            candidates = [x.format(name).lower() for x in candidates]
            if any((x[1] in candidates for x in content)):
                return True
        return False

    def search_lib_dirs(path: str) -> bool:
        for i in [os.path.join(path, x) for x in self.cmakeinfo.common_paths]:
            if not self._cached_isdir(i):
                continue
            cm_dir = os.path.join(i, 'cmake')
            if self._cached_isdir(cm_dir):
                content = self._cached_listdir(cm_dir)
                content = tuple((x for x in content if x[1].startswith(lname)))
                for k in content:
                    if find_module(os.path.join(cm_dir, k[0])):
                        return True
            content = self._cached_listdir(i)
            content = tuple((x for x in content if x[1].startswith(lname)))
            for k in content:
                if find_module(os.path.join(i, k[0])):
                    return True
        return False
    for i in module_path + [os.path.join(self.cmakeinfo.cmake_root, 'Modules')]:
        if find_module(i):
            return True
    for i in prefix_path:
        if search_lib_dirs(i):
            return True
    system_env: T.List[str] = []
    for i in os.environ.get('PATH', '').split(os.pathsep):
        if i.endswith('/bin') or i.endswith('\\bin'):
            i = i[:-4]
        if i.endswith('/sbin') or i.endswith('\\sbin'):
            i = i[:-5]
        system_env += [i]
    for i in self.cmakeinfo.module_paths + system_env:
        if find_module(i):
            return True
        if search_lib_dirs(i):
            return True
        content = self._cached_listdir(i)
        content = tuple((x for x in content if x[1].startswith(lname)))
        for k in content:
            if search_lib_dirs(os.path.join(i, k[0])):
                return True
        if machine.is_darwin():
            for j in [f'{lname}.framework', f'{lname}.app']:
                for k in content:
                    if k[1] != j:
                        continue
                    if find_module(os.path.join(i, k[0], 'Resources')) or find_module(os.path.join(i, k[0], 'Version')):
                        return True
    env_path = os.environ.get(f'{name}_DIR')
    if env_path and find_module(env_path):
        return True
    linux_reg = Path.home() / '.cmake' / 'packages'
    for p in [linux_reg / name, linux_reg / lname]:
        if p.exists():
            return True
    return False
from __future__ import annotations
import glob
import re
import os
import typing as T
from pathlib import Path
from .. import mesonlib
from .. import mlog
from ..environment import detect_cpu_family
from .base import DependencyException, SystemDependency
from .detect import packages
def _detect_arch_libdir(self) -> str:
    arch = detect_cpu_family(self.env.coredata.compilers.host)
    machine = self.env.machines[self.for_machine]
    msg = '{} architecture is not supported in {} version of the CUDA Toolkit.'
    if machine.is_windows():
        libdirs = {'x86': 'Win32', 'x86_64': 'x64'}
        if arch not in libdirs:
            raise DependencyException(msg.format(arch, 'Windows'))
        return os.path.join('lib', libdirs[arch])
    elif machine.is_linux():
        libdirs = {'x86_64': 'lib64', 'ppc64': 'lib', 'aarch64': 'lib64', 'loongarch64': 'lib64'}
        if arch not in libdirs:
            raise DependencyException(msg.format(arch, 'Linux'))
        return libdirs[arch]
    elif machine.is_darwin():
        libdirs = {'x86_64': 'lib64'}
        if arch not in libdirs:
            raise DependencyException(msg.format(arch, 'macOS'))
        return libdirs[arch]
    else:
        raise DependencyException('CUDA Toolkit: unsupported platform.')
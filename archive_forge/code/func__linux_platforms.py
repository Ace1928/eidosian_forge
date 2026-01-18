import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _linux_platforms(is_32bit: bool=_32_BIT_INTERPRETER) -> Iterator[str]:
    linux = _normalize_string(sysconfig.get_platform())
    if is_32bit:
        if linux == 'linux_x86_64':
            linux = 'linux_i686'
        elif linux == 'linux_aarch64':
            linux = 'linux_armv7l'
    _, arch = linux.split('_', 1)
    yield from _manylinux.platform_tags(linux, arch)
    yield from _musllinux.platform_tags(arch)
    yield linux
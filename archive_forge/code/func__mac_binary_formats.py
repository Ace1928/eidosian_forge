import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _mac_binary_formats(version: MacVersion, cpu_arch: str) -> List[str]:
    formats = [cpu_arch]
    if cpu_arch == 'x86_64':
        if version < (10, 4):
            return []
        formats.extend(['intel', 'fat64', 'fat32'])
    elif cpu_arch == 'i386':
        if version < (10, 4):
            return []
        formats.extend(['intel', 'fat32', 'fat'])
    elif cpu_arch == 'ppc64':
        if version > (10, 5) or version < (10, 4):
            return []
        formats.append('fat64')
    elif cpu_arch == 'ppc':
        if version > (10, 6):
            return []
        formats.extend(['fat32', 'fat'])
    if cpu_arch in {'arm64', 'x86_64'}:
        formats.append('universal2')
    if cpu_arch in {'x86_64', 'i386', 'ppc64', 'ppc', 'intel'}:
        formats.append('universal')
    return formats
import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _mac_arch(arch: str, is_32bit: bool=_32_BIT_INTERPRETER) -> str:
    if not is_32bit:
        return arch
    if arch.startswith('ppc'):
        return 'ppc'
    return 'i386'
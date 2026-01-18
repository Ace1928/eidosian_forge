import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Dict, Generator, Iterator, NamedTuple, Optional, Tuple
from ._elffile import EIClass, EIData, ELFFile, EMachine
def _is_compatible(name: str, arch: str, version: _GLibCVersion) -> bool:
    sys_glibc = _get_glibc_version()
    if sys_glibc < version:
        return False
    try:
        import _manylinux
    except ImportError:
        return True
    if hasattr(_manylinux, 'manylinux_compatible'):
        result = _manylinux.manylinux_compatible(version[0], version[1], arch)
        if result is not None:
            return bool(result)
        return True
    if version == _GLibCVersion(2, 5):
        if hasattr(_manylinux, 'manylinux1_compatible'):
            return bool(_manylinux.manylinux1_compatible)
    if version == _GLibCVersion(2, 12):
        if hasattr(_manylinux, 'manylinux2010_compatible'):
            return bool(_manylinux.manylinux2010_compatible)
    if version == _GLibCVersion(2, 17):
        if hasattr(_manylinux, 'manylinux2014_compatible'):
            return bool(_manylinux.manylinux2014_compatible)
    return True
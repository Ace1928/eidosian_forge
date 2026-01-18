import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _generic_abi() -> List[str]:
    """
    Return the ABI tag based on EXT_SUFFIX.
    """
    ext_suffix = _get_config_var('EXT_SUFFIX', warn=True)
    if not isinstance(ext_suffix, str) or ext_suffix[0] != '.':
        raise SystemError("invalid sysconfig.get_config_var('EXT_SUFFIX')")
    parts = ext_suffix.split('.')
    if len(parts) < 3:
        return _cpython_abis(sys.version_info[:2])
    soabi = parts[1]
    if soabi.startswith('cpython'):
        abi = 'cp' + soabi.split('-')[1]
    elif soabi.startswith('cp'):
        abi = soabi.split('-')[0]
    elif soabi.startswith('pypy'):
        abi = '-'.join(soabi.split('-')[:2])
    elif soabi.startswith('graalpy'):
        abi = '-'.join(soabi.split('-')[:3])
    elif soabi:
        abi = soabi
    else:
        return []
    return [_normalize_string(abi)]
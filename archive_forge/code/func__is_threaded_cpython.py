import logging
import platform
import re
import struct
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _is_threaded_cpython(abis: List[str]) -> bool:
    """
    Determine if the ABI corresponds to a threaded (`--disable-gil`) build.

    The threaded builds are indicated by a "t" in the abiflags.
    """
    if len(abis) == 0:
        return False
    m = re.match('cp\\d+(.*)', abis[0])
    if not m:
        return False
    abiflags = m.group(1)
    return 't' in abiflags
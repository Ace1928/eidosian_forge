import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _get_config_var(name: str, warn: bool=False) -> Union[int, str, None]:
    value: Union[int, str, None] = sysconfig.get_config_var(name)
    if value is None and warn:
        logger.debug("Config variable '%s' is unset, Python ABI tag may be incorrect", name)
    return value
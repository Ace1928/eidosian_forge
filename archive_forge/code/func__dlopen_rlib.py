import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _dlopen_rlib(r_home: typing.Optional[str]):
    """Open R's shared C library.

    This is only relevant in ABI mode."""
    if r_home is None:
        raise ValueError('r_home is None. Try python -m rpy2.situation')
    lib_path = rpy2.situation.get_rlib_path(r_home, platform.system())
    if lib_path is None:
        raise ValueError('The library path cannot be None.')
    else:
        rlib = ffi.dlopen(lib_path)
    return rlib
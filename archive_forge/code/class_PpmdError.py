import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
class PpmdError(Exception):
    """Call to the underlying PPMd library failed."""
    pass
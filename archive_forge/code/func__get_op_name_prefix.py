import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _get_op_name_prefix(op_name):
    """
    Check whether the given op_name starts with any words in `_OP_NAME_PREFIX_LIST`.
    If found, return the prefix; else, return an empty string.
    """
    for prefix in _OP_NAME_PREFIX_LIST:
        if op_name.startswith(prefix):
            return prefix
    return ''
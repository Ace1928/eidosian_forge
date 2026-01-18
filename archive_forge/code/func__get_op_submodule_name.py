import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _get_op_submodule_name(op_name, op_name_prefix, submodule_name_list):
    """Get the submodule name of a specific op"""
    assert op_name.startswith(op_name_prefix)
    for submodule_name in submodule_name_list:
        if op_name[len(op_name_prefix):].startswith(submodule_name):
            return submodule_name
    return ''
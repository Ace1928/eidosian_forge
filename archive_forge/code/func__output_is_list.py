import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _output_is_list(op_name):
    """ Whether the output of the operator is a list.

    Parameters
    ----------
    op_name : Name of the operator

    Returns
    -------

    """
    if _is_np_op(op_name):
        return op_name in _NP_OUTPUT_IS_LIST_OPERATORS
    return False
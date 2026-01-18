import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _find_error_type(line):
    """Find the error name given the first line of the error message.

    Parameters
    ----------
    line : str
        The first line of error message.

    Returns
    -------
    name : str The error name
    """
    end_pos = line.find(':')
    if end_pos == -1:
        return None
    err_name = line[:end_pos]
    if _valid_error_name(err_name):
        return err_name
    return None
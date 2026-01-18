import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def _is_np_op(op_name):
    return op_name.startswith(_NP_OP_PREFIX) or op_name.startswith(_NP_EXT_OP_PREFIX) or op_name.startswith(_NP_INTERNAL_OP_PREFIX)
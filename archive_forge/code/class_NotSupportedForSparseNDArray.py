import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
class NotSupportedForSparseNDArray(MXNetError):
    """Error: Not supported for SparseNDArray"""

    def __init__(self, function, alias, *args):
        super(NotSupportedForSparseNDArray, self).__init__()
        self.function = function.__name__
        self.alias = alias
        self.args = [str(type(a)) for a in args]

    def __str__(self):
        msg = 'Function {}'.format(self.function)
        if self.alias:
            msg += ' (namely operator "{}")'.format(self.alias)
        if self.args:
            msg += ' with arguments ({})'.format(', '.join(self.args))
        msg += ' is not supported for SparseNDArray and only available in NDArray.'
        return msg
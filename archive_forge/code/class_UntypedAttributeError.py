import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class UntypedAttributeError(TypingError):

    def __init__(self, value, attr, loc=None):
        module = getattr(value, 'pymod', None)
        if module is not None and module == np:
            msg = "Use of unsupported NumPy function 'numpy.%s' or unsupported use of the function." % attr
        else:
            msg = "Unknown attribute '{attr}' of type {type}"
            msg = msg.format(type=value, attr=attr)
        super(UntypedAttributeError, self).__init__(msg, loc=loc)
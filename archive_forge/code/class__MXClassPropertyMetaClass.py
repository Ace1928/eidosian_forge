import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
class _MXClassPropertyMetaClass(type):

    def __setattr__(cls, key, value):
        obj = cls.__dict__.get(key)
        if obj and isinstance(obj, _MXClassPropertyDescriptor):
            return obj.__set__(cls, value)
        return super(_MXClassPropertyMetaClass, cls).__setattr__(key, value)
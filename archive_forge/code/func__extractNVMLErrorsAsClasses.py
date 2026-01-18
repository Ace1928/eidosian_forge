from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def _extractNVMLErrorsAsClasses():
    """
    Generates a hierarchy of classes on top of NVMLError class.

    Each NVML Error gets a new NVMLError subclass. This way try,except blocks can filter appropriate
    exceptions more easily.

    NVMLError is a parent class. Each NVML_ERROR_* gets it's own subclass.
    e.g. NVML_ERROR_ALREADY_INITIALIZED will be turned into NVMLError_AlreadyInitialized
    """
    this_module = sys.modules[__name__]
    nvmlErrorsNames = [x for x in dir(this_module) if x.startswith('NVML_ERROR_')]
    for err_name in nvmlErrorsNames:
        class_name = 'NVMLError_' + string.capwords(err_name.replace('NVML_ERROR_', ''), '_').replace('_', '')
        err_val = getattr(this_module, err_name)

        def gen_new(val):

            def new(typ):
                obj = NVMLError.__new__(typ, val)
                return obj
            return new
        new_error_class = type(class_name, (NVMLError,), {'__new__': gen_new(err_val)})
        new_error_class.__module__ = __name__
        setattr(this_module, class_name, new_error_class)
        NVMLError._valClassMapping[err_val] = new_error_class
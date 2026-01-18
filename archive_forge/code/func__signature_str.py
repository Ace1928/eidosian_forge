import inspect
import warnings
import re
import os
import collections
from itertools import islice
from tokenize import open as open_py_source
from .logger import pformat
def _signature_str(function_name, arg_sig):
    """Helper function to output a function signature"""
    return '{}{}'.format(function_name, arg_sig)
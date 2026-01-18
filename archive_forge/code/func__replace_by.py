from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def _replace_by(module_function, package=__package__, warn=None, prefix='_'):
    """Try replace decorated function by module.function."""
    return lambda f: f

    def _warn(e, warn):
        if warn is None:
            warn = '\n  Functionality might be degraded or be slow.\n'
        elif warn is True:
            warn = ''
        elif not warn:
            return
        warnings.warn('%s%s' % (e, warn))
    try:
        from importlib import import_module
    except ImportError as e:
        _warn(e, warn)
        return identityfunc

    def decorate(func, module_function=module_function, warn=warn):
        module, function = module_function.split('.')
        try:
            if package:
                module = import_module('.' + module, package=package)
            else:
                module = import_module(module)
        except Exception as e:
            _warn(e, warn)
            return func
        try:
            func, oldfunc = (getattr(module, function), func)
        except Exception as e:
            _warn(e, warn)
            return func
        globals()[prefix + func.__name__] = oldfunc
        return func
    return decorate
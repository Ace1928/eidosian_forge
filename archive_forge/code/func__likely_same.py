import importlib
import logging
import os
import sys
def _likely_same(a, b):
    try:
        if sys.platform == 'win32':
            if os.stat(a) == os.stat(b):
                return True
        elif os.path.samefile(a, b):
            return True
    except OSError:
        return False
    if chop_py_suffix(a) == chop_py_suffix(b):
        return True
    return False
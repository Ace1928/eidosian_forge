import re
import functools
import distutils.core
import distutils.errors
import distutils.extension
from .monkey import get_unpatched
def _have_cython():
    """
    Return True if Cython can be imported.
    """
    cython_impl = 'Cython.Distutils.build_ext'
    try:
        __import__(cython_impl, fromlist=['build_ext']).build_ext
        return True
    except Exception:
        pass
    return False
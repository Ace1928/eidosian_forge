import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
class VectorOperationsDelegator(object):
    """
    Delegate operations such as __getitem__, __add__, etc...
    to the corresponding R function.
    This permits a convenient coexistence between
    operators on Python sequence object with their R conterparts.
    """

    def __init__(self, parent):
        """ The parent in expected to inherit from Vector. """
        self._parent = parent

    def __add__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('+')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __sub__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('-')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __matmul__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('%*%')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __mul__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('*')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __pow__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('^')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __floordiv__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('%/%')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __truediv__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('/')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __mod__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('%%')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __or__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('|')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __and__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('&')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __invert__(self):
        res = globalenv_ri.find('!')(self._parent)
        return conversion.get_conversion().rpy2py(res)

    def __lt__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('<')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __le__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('<=')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __eq__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('==')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __ne__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('!=')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __gt__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('>')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __ge__(self, x):
        cv = conversion.get_conversion()
        res = globalenv_ri.find('>=')(self._parent, cv.py2rpy(x))
        return cv.rpy2py(res)

    def __neg__(self):
        res = globalenv_ri.find('-')(self._parent)
        return res

    def __contains__(self, what):
        res = globalenv_ri.find('%in%')(what, self._parent)
        return res[0]
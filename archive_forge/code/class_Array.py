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
class Array(Vector):
    """ An R array """
    _dimnames_get = baseenv_ri['dimnames']
    _dimnames_set = baseenv_ri['dimnames<-']
    _dim_get = baseenv_ri['dim']
    _dim_set = baseenv_ri['dim<-']
    _isarray = baseenv_ri['is.array']

    def __dim_get(self):
        res = self._dim_get(self)
        res = conversion.get_conversion().rpy2py(res)
        return res

    def __dim_set(self, value):
        raise NotImplementedError('Not yet implemented')
        value = conversion.get_conversion().py2rpy(value)
        self._dim_set(self, value)
    dim = property(__dim_get, __dim_set, None, 'Get or set the dimension of the array.')

    @property
    def names(self) -> sexp.Sexp:
        """ Return a list of name vectors
        (like the R function 'dimnames' does)."""
        res = self._dimnames_get(self)
        res = conversion.get_conversion().rpy2py(res)
        return res

    @names.setter
    def names(self, value) -> None:
        """ Set list of name vectors
        (like the R function 'dimnames' does)."""
        value = conversion.get_conversion().rpy2py(value)
        res = self._dimnames_set(self, value)
        self.__sexp__ = res.__sexp__
    dimnames = names
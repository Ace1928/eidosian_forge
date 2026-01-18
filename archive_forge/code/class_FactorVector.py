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
class FactorVector(IntVector):
    """ Vector of 'factors'.

    FactorVector(obj,
                 levels = rinterface.MissingArg,
                 labels = rinterface.MissingArg,
                 exclude = rinterface.MissingArg,
                 ordered = rinterface.MissingArg) -> FactorVector

    obj: StrVector or StrSexpVector
    levels: StrVector or StrSexpVector
    labels: StrVector or StrSexpVector (of same length as levels)
    exclude: StrVector or StrSexpVector
    ordered: boolean

    """
    _factor = baseenv_ri['factor']
    _levels = baseenv_ri['levels']
    _levels_set = baseenv_ri['levels<-']
    _nlevels = baseenv_ri['nlevels']
    _isordered = baseenv_ri['is.ordered']
    NAvalue = rinterface.NA_Integer

    def __init__(self, obj, levels=rinterface.MissingArg, labels=rinterface.MissingArg, exclude=rinterface.MissingArg, ordered=rinterface.MissingArg):
        if not isinstance(obj, Sexp):
            obj = StrSexpVector(obj)
        if 'factor' in obj.rclass and all((p is rinterface.MissingArg for p in (labels, exclude, ordered))):
            res = obj
        else:
            res = self._factor(obj, levels=levels, labels=labels, exclude=exclude, ordered=ordered)
        super(FactorVector, self).__init__(res)

    def repr_format_elt(self, elt, max_width=8):
        max_width = int(max_width)
        levels = self._levels(self)
        if elt is NA_Integer:
            res = repr(elt)
        else:
            res = levels[elt - 1]
            if len(res) >= max_width:
                res = '%s...' % res[:max_width - 3]
        return res

    def __levels_get(self):
        res = self._levels(self)
        return conversion.rpy2py(res)

    def __levels_set(self, value):
        res = self._levels_set(self, conversion.get_conversion().py2rpy(value))
        self.__sexp__ = res.__sexp__
    levels = property(__levels_get, __levels_set)

    def __nlevels_get(self):
        res = self._nlevels(self)
        return res[0]
    nlevels = property(__nlevels_get, None, None, 'number of levels ')

    def __isordered_get(self):
        res = self._isordered(self)
        return res[0]
    isordered = property(__isordered_get, None, None, 'are the levels in the factor ordered ?')

    def iter_labels(self):
        """ Iterate the over the labels, that is iterate over
        the items returning associated label for each item """
        levels = self.levels
        for x in conversion.noconversion(self):
            yield (rinterface.NA_Character if x is rinterface.NA_Integer else levels[x - 1])
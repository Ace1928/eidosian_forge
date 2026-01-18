from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
class Units(Unit):
    """ Units type for values with physical units """
    cls = MEASURE
    __slots__ = ('unit', 'tp')

    def __init__(self, unit, tp=None):
        if not isinstance(unit, str):
            raise TypeError('unit parameter to units datashape must be a string')
        if tp is None:
            tp = DataShape(float64)
        elif not isinstance(tp, DataShape):
            raise TypeError('tp parameter to units datashape must be a datashape type')
        self.unit = unit
        self.tp = tp

    def __str__(self):
        if self.tp == DataShape(float64):
            return 'units[%r]' % self.unit
        else:
            return 'units[%r, %s]' % (self.unit, self.tp)
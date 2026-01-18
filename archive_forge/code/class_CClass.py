import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
class CClass(AxisConcatenator):

    def __init__(self):
        super(CClass, self).__init__(-1, ndmin=2, trans1d=0)
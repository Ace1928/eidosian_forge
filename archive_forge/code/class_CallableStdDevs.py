from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
class CallableStdDevs(numpy.matrix):
    """
    Class for standard deviation results, which used to be
    callable. Provided for compatibility with old code. Issues an
    obsolescence warning upon call.

    New objects must be created by passing an existing
    """

    def __new__(cls, matrix):
        matrix.__class__ = cls
        return matrix

    def __call__(self):
        deprecation('the std_devs attribute should not be called anymore: use .std_devs instead of .std_devs().')
        return self
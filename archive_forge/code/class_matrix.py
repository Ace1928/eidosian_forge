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
class matrix(numpy.matrix):
    """
    Class equivalent to numpy.matrix, but that behaves better when the
    matrix contains numbers with uncertainties.
    """

    def __rmul__(self, other):
        if numeric.isscalar(other):
            return numeric.dot(self, other)
        else:
            return numeric.dot(other, self)

    def getI(self):
        """Matrix inverse or pseudo-inverse."""
        m, n = self.shape
        return (inv if m == n else pinv)(self)
    I = numpy.matrix.I.getter(getI)

    @property
    def nominal_values(self):
        """
        Nominal value of all the elements of the matrix.
        """
        return nominal_values(self)

    @property
    def std_devs(self):
        return CallableStdDevs(std_devs(self))
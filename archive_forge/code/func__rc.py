from numpy.core import (
def _rc(self, a):
    if len(shape(a)) == 0:
        return a
    else:
        return self.__class__(a)
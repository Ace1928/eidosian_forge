from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
def _getm(method):
    """
    Return a function to calculate m for Illinois-like methods.
    """
    if method == 'illinois':

        def getm(fz, fb):
            return 0.5
    elif method == 'pegasus':

        def getm(fz, fb):
            return fb / (fb + fz)
    elif method == 'anderson':

        def getm(fz, fb):
            m = 1 - fz / fb
            if m > 0:
                return m
            else:
                return 0.5
    else:
        raise ValueError("method '%s' not recognized" % method)
    return getm
import math
import numpy
def R2(orig, residSum):
    """ returns the R2 value for a set of predictions """
    vect = numpy.array(orig)
    n = vect.shape[0]
    if n <= 0:
        return (0.0, 0.0)
    oMean = sum(vect) / n
    v = vect - oMean
    oVar = sum(v * v)
    return 1.0 - residSum / oVar
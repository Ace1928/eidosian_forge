import numpy
from . import ClusterUtils
def _scaleMetric(val, power=2, min=0.0001):
    val = float(val)
    nval = pow(val, power)
    if nval < min:
        return 0.0
    else:
        return numpy.log(nval / min)
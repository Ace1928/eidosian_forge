from scipy._lib.uarray import generate_multimethod, Dispatchable
import numpy as np
def _x_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the transform input array (``x``)
    """
    if len(args) > 0:
        return ((dispatchables[0],) + args[1:], kwargs)
    kw = kwargs.copy()
    kw['x'] = dispatchables[0]
    return (args, kw)
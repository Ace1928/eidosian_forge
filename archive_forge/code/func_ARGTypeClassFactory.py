import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect
def ARGTypeClassFactory(classname, prange, BaseClass=ARGType):
    """Dynamically create parameter type class.

    Parameters
    ----------
    classname: string
        parameter name in a operator
    prange: list
        list of values for the parameter in a operator
    BaseClass: Class
        inherited BaseClass for parameter

    Returns
    -------
    Class
        parameter class

    """
    return type(classname, (BaseClass,), {'values': prange})
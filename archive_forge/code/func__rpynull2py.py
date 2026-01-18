from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
def _rpynull2py(robject):
    if robject is rpy2.rinterface.NULL:
        robject = None
    return robject
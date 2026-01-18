from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
def _rpysce2py(robject):
    if utils._try_import('anndata2ri'):
        robject = anndata2ri.rpy2py(robject)
        if hasattr(robject, 'uns'):
            for k, v in robject.uns.items():
                robject.uns[k] = rpy2py(v)
    return robject
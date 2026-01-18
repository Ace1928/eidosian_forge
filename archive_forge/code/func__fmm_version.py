import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def _fmm_version():
    from . import _fmm_core
    return _fmm_core.__version__
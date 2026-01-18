import numpy as np
from warnings import warn
from ._optimize import MemoizeJac, OptimizeResult, _check_unknown_options
from ._minpack_py import _root_hybr, leastsq
from ._spectral import _root_df_sane
from . import _nonlin as nonlin
def _warn_jac_unused(jac, method):
    if jac is not None:
        warn(f'Method {method} does not use the jacobian (jac).', RuntimeWarning, stacklevel=2)
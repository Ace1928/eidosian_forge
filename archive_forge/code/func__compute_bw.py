import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def _compute_bw(self, bw):
    """
        Computes the bandwidth of the data.

        Parameters
        ----------
        bw : {array_like, str}
            If array_like: user-specified bandwidth.
            If a string, should be one of:

                - cv_ml: cross validation maximum likelihood
                - normal_reference: normal reference rule of thumb
                - cv_ls: cross validation least squares

        Notes
        -----
        The default values for bw is 'normal_reference'.
        """
    if bw is None:
        bw = 'normal_reference'
    if not isinstance(bw, str):
        self._bw_method = 'user-specified'
        res = np.asarray(bw)
    else:
        self._bw_method = bw
        if bw == 'normal_reference':
            bwfunc = self._normal_reference
        elif bw == 'cv_ml':
            bwfunc = self._cv_ml
        else:
            bwfunc = self._cv_ls
        res = bwfunc()
    return res
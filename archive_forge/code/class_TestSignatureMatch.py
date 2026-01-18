import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
class TestSignatureMatch:
    NANFUNCS = {np.nanmin: np.amin, np.nanmax: np.amax, np.nanargmin: np.argmin, np.nanargmax: np.argmax, np.nansum: np.sum, np.nanprod: np.prod, np.nancumsum: np.cumsum, np.nancumprod: np.cumprod, np.nanmean: np.mean, np.nanmedian: np.median, np.nanpercentile: np.percentile, np.nanquantile: np.quantile, np.nanvar: np.var, np.nanstd: np.std}
    IDS = [k.__name__ for k in NANFUNCS]

    @staticmethod
    def get_signature(func, default='...'):
        """Construct a signature and replace all default parameter-values."""
        prm_list = []
        signature = inspect.signature(func)
        for prm in signature.parameters.values():
            if prm.default is inspect.Parameter.empty:
                prm_list.append(prm)
            else:
                prm_list.append(prm.replace(default=default))
        return inspect.Signature(prm_list)

    @pytest.mark.parametrize('nan_func,func', NANFUNCS.items(), ids=IDS)
    def test_signature_match(self, nan_func, func):
        signature = self.get_signature(func)
        nan_signature = self.get_signature(nan_func)
        np.testing.assert_equal(signature, nan_signature)

    def test_exhaustiveness(self):
        """Validate that all nan functions are actually tested."""
        np.testing.assert_equal(set(self.IDS), set(np.lib.nanfunctions.__all__))
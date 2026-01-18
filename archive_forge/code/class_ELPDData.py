import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
class ELPDData(pd.Series):
    """Class to contain the data from elpd information criterion like waic or loo."""

    def __str__(self):
        """Print elpd data in a user friendly way."""
        kind = self.index[0].split('_')[1]
        if kind not in ('loo', 'waic'):
            raise ValueError('Invalid ELPDData object')
        scale_str = SCALE_DICT[self['scale']]
        padding = len(scale_str) + len(kind) + 1
        base = BASE_FMT.format(padding, padding - 2)
        base = base.format('', *self.values, kind=kind, scale=scale_str, n_samples=self.n_samples, n_points=self.n_data_points)
        if self.warning:
            base += '\n\nThere has been a warning during the calculation. Please check the results.'
        if kind == 'loo' and 'pareto_k' in self:
            bins = np.asarray([-np.inf, 0.5, 0.7, 1, np.inf])
            counts, *_ = _histogram(self.pareto_k.values, bins)
            extended = POINTWISE_LOO_FMT.format(max(4, len(str(np.max(counts)))))
            extended = extended.format('Count', 'Pct.', *[*counts, *counts / np.sum(counts) * 100])
            base = '\n'.join([base, extended])
        return base

    def __repr__(self):
        """Alias to ``__str__``."""
        return self.__str__()

    def copy(self, deep=True):
        """Perform a pandas deep copy of the ELPDData plus a copy of the stored data."""
        copied_obj = pd.Series.copy(self)
        for key in copied_obj.keys():
            if deep:
                copied_obj[key] = _deepcopy(copied_obj[key])
            else:
                copied_obj[key] = _copy(copied_obj[key])
        return ELPDData(copied_obj)
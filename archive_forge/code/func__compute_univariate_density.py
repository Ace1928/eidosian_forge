from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import (
from .utils import (
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import (
important parameter. Misspecification of the bandwidth can produce a
def _compute_univariate_density(self, data_variable, common_norm, common_grid, estimate_kws, warn_singular=True):
    estimator = KDE(**estimate_kws)
    if set(self.variables) - {'x', 'y'}:
        if common_grid:
            all_observations = self.comp_data.dropna()
            estimator.define_support(all_observations[data_variable])
    else:
        common_norm = False
    all_data = self.plot_data.dropna()
    if common_norm and 'weights' in all_data:
        whole_weight = all_data['weights'].sum()
    else:
        whole_weight = len(all_data)
    densities = {}
    for sub_vars, sub_data in self.iter_data('hue', from_comp_data=True):
        observations = sub_data[data_variable]
        if 'weights' in self.variables:
            weights = sub_data['weights']
            part_weight = weights.sum()
        else:
            weights = None
            part_weight = len(sub_data)
        variance = np.nan_to_num(observations.var())
        singular = len(observations) < 2 or math.isclose(variance, 0)
        try:
            if not singular:
                density, support = estimator(observations, weights=weights)
        except np.linalg.LinAlgError:
            singular = True
        if singular:
            msg = 'Dataset has 0 variance; skipping density estimate. Pass `warn_singular=False` to disable this warning.'
            if warn_singular:
                warnings.warn(msg, UserWarning, stacklevel=4)
            continue
        _, f_inv = self._get_scale_transforms(self.data_variable)
        support = f_inv(support)
        if common_norm:
            density *= part_weight / whole_weight
        key = tuple(sub_vars.items())
        densities[key] = pd.Series(density, index=support)
    return densities
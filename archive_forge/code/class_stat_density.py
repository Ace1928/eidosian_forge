from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.evaluation import after_stat
from .stat import stat
@document
class stat_density(stat):
    """
    Compute density estimate

    {usage}

    Parameters
    ----------
    {common_parameters}
    kernel : str, default="gaussian"
        Kernel used for density estimation. One of:
        ```python
        "biweight"
        "cosine"
        "cosine2"
        "epanechnikov"
        "gaussian"
        "triangular"
        "triweight"
        "uniform"
        ```
    adjust : float, default=1
        An adjustment factor for the `bw`. Bandwidth becomes
        `bw * adjust`{.py}.
        Adjustment of the bandwidth.
    trim : bool, default=False
        This parameter only matters if you are displaying multiple
        densities in one plot. If `False`{.py}, the default, each
        density is computed on the full range of the data. If
        `True`{.py}, each density is computed over the range of that
        group; this typically means the estimated x values will not
        line-up, and hence you won't be able to stack density values.
    n : int, default=1024
        Number of equally spaced points at which the density is to
        be estimated. For efficient computation, it should be a power
        of two.
    gridsize : int, default=None
        If gridsize is `None`{.py}, `max(len(x), 50)`{.py} is used.
    bw : str | float, default="nrd0"
        The bandwidth to use, If a float is given, it is the bandwidth.
        The options are:

        ```python
        "nrd0"
        "normal_reference"
        "scott"
        "silverman"
        ```

        `nrd0` is a port of `stats::bw.nrd0` in R; it is eqiuvalent
        to `silverman` when there is more than 1 value in a group.
    cut : float, default=3
        Defines the length of the grid past the lowest and highest
        values of `x` so that the kernel goes to zero. The end points
        are `-/+ cut*bw*{min(x) or max(x)}`.
    clip : tuple[float, float], default=(-inf, inf)
        Values in `x` that are outside of the range given by clip are
        dropped. The number of values in `x` is then shortened.

    See Also
    --------
    plotnine.geom_density
    statsmodels.nonparametric.kde.KDEUnivariate
    statsmodels.nonparametric.kde.KDEUnivariate.fit
    """
    _aesthetics_doc = "\n    {aesthetics_table}\n\n    **Options for computed aesthetics**\n\n    ```python\n    'density'   # density estimate\n\n    'count'     # density * number of points,\n                # useful for stacked density plots\n\n    'scaled'    # density estimate, scaled to maximum of 1\n    ```\n\n        'n'         # Number of observations at a position\n\n    "
    REQUIRED_AES = {'x'}
    DEFAULT_PARAMS = {'geom': 'density', 'position': 'stack', 'na_rm': False, 'kernel': 'gaussian', 'adjust': 1, 'trim': False, 'n': 1024, 'gridsize': None, 'bw': 'nrd0', 'cut': 3, 'clip': (-np.inf, np.inf)}
    DEFAULT_AES = {'y': after_stat('density')}
    CREATES = {'density', 'count', 'scaled', 'n'}

    def setup_params(self, data):
        params = self.params.copy()
        lookup = {'biweight': 'biw', 'cosine': 'cos', 'cosine2': 'cos2', 'epanechnikov': 'epa', 'gaussian': 'gau', 'triangular': 'tri', 'triweight': 'triw', 'uniform': 'uni'}
        with suppress(KeyError):
            params['kernel'] = lookup[params['kernel'].lower()]
        if params['kernel'] not in lookup.values():
            msg = f'kernel should be one of {lookup.keys()}. You may use the abbreviations {lookup.values()}'
            raise PlotnineError(msg)
        return params

    @classmethod
    def compute_group(cls, data, scales, **params):
        weight = data.get('weight')
        if params['trim']:
            range_x = (data['x'].min(), data['x'].max())
        else:
            range_x = scales.x.dimension()
        return compute_density(data['x'], weight, range_x, **params)
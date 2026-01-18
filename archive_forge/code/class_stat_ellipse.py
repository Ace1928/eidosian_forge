from __future__ import annotations
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineWarning
from .stat import stat
@document
class stat_ellipse(stat):
    """
    Calculate normal confidence interval ellipse

    {usage}

    Parameters
    ----------
    {common_parameters}
    type : Literal["t", "norm", "euclid"], default="t"
        The type of ellipse.
        `t` assumes a multivariate t-distribution.
        `norm` assumes a multivariate normal distribution.
        `euclid` draws a circle with the radius equal to
        `level`, representing the euclidean distance from the center.

    level : float, default=0.95
        The confidence level at which to draw the ellipse.
    segments : int, default=51
        Number of segments to be used in drawing the ellipse.
    """
    REQUIRED_AES = {'x', 'y'}
    DEFAULT_PARAMS = {'geom': 'path', 'position': 'identity', 'na_rm': False, 'type': 't', 'level': 0.95, 'segments': 51}

    @classmethod
    def compute_group(cls, data, scales, **params):
        import scipy.stats as stats
        from scipy import linalg
        level = params['level']
        segments = params['segments']
        type_ = params['type']
        dfn = 2
        dfd = len(data) - 1
        if dfd < 3:
            warn('Too few points to calculate an ellipse', PlotnineWarning)
            return pd.DataFrame({'x': [], 'y': []})
        m: FloatArray = np.asarray(data[['x', 'y']])
        if type_ == 't':
            res = cov_trob(m)
            cov = res['cov']
            center = res['center']
        elif type_ == 'norm':
            cov = np.cov(m, rowvar=False)
            center = np.mean(m, axis=0)
        elif type_ == 'euclid':
            cov = np.cov(m, rowvar=False)
            cov = np.diag(np.repeat(np.diag(cov).min(), 2))
            center = np.mean(m, axis=0)
        else:
            raise ValueError(f'Unknown value for type={type_}')
        chol_decomp = linalg.cholesky(cov, lower=False)
        if type_ == 'euclid':
            radius = level / chol_decomp.max()
        else:
            radius = np.sqrt(dfn * stats.f.ppf(level, dfn, dfd))
        space = np.linspace(0, 2 * np.pi, segments)
        unit_circle = np.column_stack([np.cos(space), np.sin(space)])
        res = center + radius * np.dot(unit_circle, chol_decomp)
        return pd.DataFrame({'x': res[:, 0], 'y': res[:, 1]})
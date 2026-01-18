import numpy as np
import pandas as pd
from ..doctools import document
from .density import get_var_type, kde
from .stat import stat
@document
class stat_density_2d(stat):
    """
    Compute 2D kernel density estimation

    {usage}

    Parameters
    ----------
    {common_parameters}
    contour : bool, default=True
        Whether to create contours of the 2d density estimate.
    n : int, default=64
        Number of equally spaced points at which the density is to
        be estimated. For efficient computation, it should be a power
        of two.
    levels : int | array_like, default=5
        Contour levels. If an integer, it specifies the maximum number
        of levels, if array_like it is the levels themselves.
    package : Literal["statsmodels", "scipy", "sklearn"], default="statsmodels"
        Package whose kernel density estimation to use.
    kde_params : dict
        Keyword arguments to pass on to the kde class.

    See Also
    --------
    statsmodels.nonparametric.kernel_density.KDEMultivariate
    scipy.stats.gaussian_kde
    sklearn.neighbors.KernelDensity
    """
    _aesthetics_doc = '\n    {aesthetics_table}\n\n    **Options for computed aesthetics**\n\n    ```python\n    "level"     # density level of a contour\n    "density"   # Computed density at a point\n    "piece"     # Numeric id of a contour in a given group\n    ```\n\n    `level` is only relevant when contours are computed. `density`\n    is available only when no contours are computed. `piece` is\n    largely irrelevant.\n    '
    REQUIRED_AES = {'x'}
    DEFAULT_PARAMS = {'geom': 'density_2d', 'position': 'identity', 'na_rm': False, 'contour': True, 'package': 'statsmodels', 'kde_params': None, 'n': 64, 'levels': 5}
    CREATES = {'y'}

    def setup_params(self, data):
        params = self.params.copy()
        if params['kde_params'] is None:
            params['kde_params'] = {}
        kde_params = params['kde_params']
        if params['package'] == 'statsmodels':
            params['package'] = 'statsmodels-m'
            if 'var_type' not in kde_params:
                x_type = get_var_type(data['x'])
                y_type = get_var_type(data['y'])
                kde_params['var_type'] = f'{x_type}{y_type}'
        return params

    @classmethod
    def compute_group(cls, data, scales, **params):
        package = params['package']
        kde_params = params['kde_params']
        group = data['group'].iloc[0]
        range_x = scales.x.dimension()
        range_y = scales.y.dimension()
        x = np.linspace(range_x[0], range_x[1], params['n'])
        y = np.linspace(range_y[0], range_y[1], params['n'])
        X, Y = np.meshgrid(x, y)
        var_data = np.array([data['x'].to_numpy(), data['y'].to_numpy()]).T
        grid = np.array([X.flatten(), Y.flatten()]).T
        density = kde(var_data, grid, package, **kde_params)
        if params['contour']:
            Z = density.reshape(len(x), len(y))
            data = contour_lines(X, Y, Z, params['levels'])
            groups = str(group) + '-00' + data['piece'].astype(str)
            data['group'] = groups
        else:
            data = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'density': density.flatten(), 'group': group, 'level': 1, 'piece': 1})
        return data
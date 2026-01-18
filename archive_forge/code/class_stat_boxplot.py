import numpy as np
import pandas as pd
from .._utils import resolution
from ..doctools import document
from .stat import stat
@document
class stat_boxplot(stat):
    """
    Compute boxplot statistics

    {usage}

    Parameters
    ----------
    {common_parameters}
    coef : float, default=1.5
        Length of the whiskers as a multiple of the Interquartile
        Range.

    See Also
    --------
    plotnine.geom_boxplot
    """
    _aesthetics_doc = '\n    {aesthetics_table}\n\n    **Options for computed aesthetics**\n\n    ```python\n    "width"  # width of boxplot\n    "lower"  # lower hinge, 25% quantile\n    "middle" # median, 50% quantile\n    "upper"  # upper hinge, 75% quantile\n\n    # lower edge of notch, computed as;\n    # median - 1.58 * IQR / sqrt(n)\n    "notchlower"\n\n    # upper edge of notch, computed as;\n    # median + 1.58 * IQR / sqrt(n)\n    "notchupper"\n\n    # lower whisker, computed as; smallest observation\n    # greater than or equal to lower hinge - 1.5 * IQR\n    "ymin"\n\n    # upper whisker, computed as; largest observation\n    # less than or equal to upper hinge + 1.5 * IQR\n    "ymax"\n    ```\n\n        \'n\'     # Number of observations at a position\n\n    Calculated aesthetics are accessed using the `after_stat` function.\n    e.g. `after_stat(\'width\')`{.py}.\n    '
    REQUIRED_AES = {'x', 'y'}
    NON_MISSING_AES = {'weight'}
    DEFAULT_PARAMS = {'geom': 'boxplot', 'position': 'dodge', 'na_rm': False, 'coef': 1.5, 'width': None}
    CREATES = {'lower', 'upper', 'middle', 'ymin', 'ymax', 'outliers', 'notchupper', 'notchlower', 'width', 'relvarwidth', 'n'}

    def setup_data(self, data):
        if 'x' not in data:
            data['x'] = 0
        return data

    def setup_params(self, data):
        if self.params['width'] is None:
            x = data.get('x', 0)
            self.params['width'] = resolution(x, False) * 0.75
        return self.params

    @classmethod
    def compute_group(cls, data, scales, **params):
        n = len(data)
        y = data['y'].to_numpy()
        if 'weight' in data:
            weights = data['weight']
            total_weight = np.sum(weights)
        else:
            weights = None
            total_weight = len(y)
        res = weighted_boxplot_stats(y, weights=weights, whis=params['coef'])
        if len(np.unique(data['x'])) > 1:
            width = np.ptp(data['x']) * 0.9
        else:
            width = params['width']
        if isinstance(data['x'].dtype, pd.CategoricalDtype):
            x = data['x'].iloc[0]
        else:
            x = np.mean([data['x'].min(), data['x'].max()])
        d = {'ymin': res['whislo'], 'lower': res['q1'], 'middle': [res['med']], 'upper': res['q3'], 'ymax': res['whishi'], 'outliers': [res['fliers']], 'notchupper': res['cihi'], 'notchlower': res['cilo'], 'x': x, 'width': width, 'relvarwidth': np.sqrt(total_weight), 'n': n}
        return pd.DataFrame(d)
import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
def _plot_index(self, y, ylabel, threshold=None, title=None, ax=None, **kwds):
    from statsmodels.graphics import utils
    fig, ax = utils.create_mpl_ax(ax)
    if title is None:
        title = 'Index Plot'
    nobs = len(self.endog)
    index = np.arange(nobs)
    ax.scatter(index, y, **kwds)
    if threshold == 'all':
        large_points = np.ones(nobs, np.bool_)
    else:
        large_points = np.abs(y) > threshold
    psize = 3 * np.ones(nobs)
    labels = self.results.model.data.row_labels
    if labels is None:
        labels = np.arange(nobs)
    ax = utils.annotate_axes(np.where(large_points)[0], labels, lzip(index, y), lzip(-psize, psize), 'large', ax)
    font = {'fontsize': 16, 'color': 'black'}
    ax.set_ylabel(ylabel, **font)
    ax.set_xlabel('Observation', **font)
    ax.set_title(title, **font)
    return fig
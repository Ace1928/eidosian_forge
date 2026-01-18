import numbers
from itertools import chain
from math import ceil
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from ...base import is_regressor
from ...utils import (
from ...utils._encode import _unique
from ...utils.parallel import Parallel, delayed
from .. import partial_dependence
from .._pd_utils import _check_feature_names, _get_feature_index
def _plot_one_way_partial_dependence(self, kind, preds, avg_preds, feature_values, feature_idx, n_ice_lines, ax, n_cols, pd_plot_idx, n_lines, ice_lines_kw, pd_line_kw, categorical, bar_kw, pdp_lim):
    """Plot 1-way partial dependence: ICE and PDP.

        Parameters
        ----------
        kind : str
            The kind of partial plot to draw.
        preds : ndarray of shape                 (n_instances, n_grid_points) or None
            The predictions computed for all points of `feature_values` for a
            given feature for all samples in `X`.
        avg_preds : ndarray of shape (n_grid_points,)
            The average predictions for all points of `feature_values` for a
            given feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        feature_idx : int
            The index corresponding to the target feature.
        n_ice_lines : int
            The number of ICE lines to plot.
        ax : Matplotlib axes
            The axis on which to plot the ICE and PDP lines.
        n_cols : int or None
            The number of column in the axis.
        pd_plot_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        n_lines : int
            The total number of lines expected to be plot on the axis.
        ice_lines_kw : dict
            Dict with keywords passed when plotting the ICE lines.
        pd_line_kw : dict
            Dict with keywords passed when plotting the PD plot.
        categorical : bool
            Whether feature is categorical.
        bar_kw: dict
            Dict with keywords passed when plotting the PD bars (categorical).
        pdp_lim : dict
            Global min and max average predictions, such that all plots will
            have the same scale and y limits. `pdp_lim[1]` is the global min
            and max for single partial dependence curves.
        """
    from matplotlib import transforms
    if kind in ('individual', 'both'):
        self._plot_ice_lines(preds[self.target_idx], feature_values, n_ice_lines, ax, pd_plot_idx, n_lines, ice_lines_kw)
    if kind in ('average', 'both'):
        if kind == 'average':
            pd_line_idx = pd_plot_idx
        else:
            pd_line_idx = pd_plot_idx * n_lines + n_ice_lines
        self._plot_average_dependence(avg_preds[self.target_idx].ravel(), feature_values, ax, pd_line_idx, pd_line_kw, categorical, bar_kw)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    vlines_idx = np.unravel_index(pd_plot_idx, self.deciles_vlines_.shape)
    if self.deciles.get(feature_idx[0], None) is not None:
        self.deciles_vlines_[vlines_idx] = ax.vlines(self.deciles[feature_idx[0]], 0, 0.05, transform=trans, color='k')
    min_val = min((val[0] for val in pdp_lim.values()))
    max_val = max((val[1] for val in pdp_lim.values()))
    ax.set_ylim([min_val, max_val])
    if not ax.get_xlabel():
        ax.set_xlabel(self.feature_names[feature_idx[0]])
    if n_cols is None or pd_plot_idx % n_cols == 0:
        if not ax.get_ylabel():
            ax.set_ylabel('Partial dependence')
    else:
        ax.set_yticklabels([])
    if pd_line_kw.get('label', None) and kind != 'individual' and (not categorical):
        ax.legend()
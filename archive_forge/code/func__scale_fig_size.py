import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def _scale_fig_size(figsize, textsize, rows=1, cols=1):
    """Scale figure properties according to rows and cols.

    Parameters
    ----------
    figsize : float or None
        Size of figure in inches
    textsize : float or None
        fontsize
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    figsize : float or None
        Size of figure in inches
    ax_labelsize : int
        fontsize for axes label
    titlesize : int
        fontsize for title
    xt_labelsize : int
        fontsize for axes ticks
    linewidth : int
        linewidth
    markersize : int
        markersize
    """
    params = mpl.rcParams
    rc_width, rc_height = tuple(params['figure.figsize'])
    rc_ax_labelsize = params['axes.labelsize']
    rc_titlesize = params['axes.titlesize']
    rc_xt_labelsize = params['xtick.labelsize']
    rc_linewidth = params['lines.linewidth']
    rc_markersize = params['lines.markersize']
    if isinstance(rc_ax_labelsize, str):
        rc_ax_labelsize = 15
    if isinstance(rc_titlesize, str):
        rc_titlesize = 16
    if isinstance(rc_xt_labelsize, str):
        rc_xt_labelsize = 14
    if figsize is None:
        width, height = (rc_width, rc_height)
        sff = 1 if rows == cols == 1 else 1.15
        width = width * cols * sff
        height = height * rows * sff
    else:
        width, height = figsize
    if textsize is not None:
        scale_factor = textsize / rc_xt_labelsize
    elif rows == cols == 1:
        scale_factor = (width * height / (rc_width * rc_height)) ** 0.5
    else:
        scale_factor = 1
    ax_labelsize = rc_ax_labelsize * scale_factor
    titlesize = rc_titlesize * scale_factor
    xt_labelsize = rc_xt_labelsize * scale_factor
    linewidth = rc_linewidth * scale_factor
    markersize = rc_markersize * scale_factor
    return ((width, height), ax_labelsize, titlesize, xt_labelsize, linewidth, markersize)
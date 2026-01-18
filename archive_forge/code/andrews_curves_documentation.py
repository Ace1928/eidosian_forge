import numpy as np
import pandas as pd
import holoviews as hv
import colorcet as cc
from ..backend_transforms import _transfer_opts_cur_backend
from ..util import with_hv_extension

    Generate a plot of Andrews curves, for visualising clusters of
    multivariate data.

    Andrews curves have the functional form:

    f(t) = x_1/sqrt(2) + x_2 sin(t) + x_3 cos(t) +
           x_4 sin(2t) + x_5 cos(2t) + ...

    Where x coefficients correspond to the values of each dimension and t is
    linearly spaced between -pi and +pi. Each row of frame then corresponds to
    a single curve.

    Parameters
    ----------
    frame: DataFrame
        Data to be plotted, preferably normalized to (0.0, 1.0)
    class_column: str
        Column name containing class names
    samples: int, optional
        Number of samples to draw
    alpha: float, optional
        The transparency of the lines
    cmap/colormap: str or colormap object
        Colormap to use for groups

    Returns
    -------
    obj : HoloViews object
        The HoloViews representation of the plot.

    See Also
    --------
    pandas.plotting.parallel_coordinates : matplotlib version of this routine
    
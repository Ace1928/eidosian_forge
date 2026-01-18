from numbers import Number
import functools
from types import MethodType
import numpy as np
from matplotlib import _api, cbook
from matplotlib.gridspec import SubplotSpec
from .axes_divider import Size, SubplotDivider, Divider
from .mpl_axes import Axes, SimpleAxisArtist

        Parameters
        ----------
        fig : `.Figure`
            The parent figure.
        rect : (float, float, float, float) or int
            The axes position, as a ``(left, bottom, width, height)`` tuple or
            as a three-digit subplot position code (e.g., "121").
        nrows_ncols : (int, int)
            Number of rows and columns in the grid.
        ngrids : int or None, default: None
            If not None, only the first *ngrids* axes in the grid are created.
        direction : {"row", "column"}, default: "row"
            Whether axes are created in row-major ("row by row") or
            column-major order ("column by column").  This also affects the
            order in which axes are accessed using indexing (``grid[index]``).
        axes_pad : float or (float, float), default: 0.02in
            Padding or (horizontal padding, vertical padding) between axes, in
            inches.
        share_all : bool, default: False
            Whether all axes share their x- and y-axis.  Note that in any case,
            all axes in a column share their x-axis and all axes in a row share
            their y-axis.
        aspect : bool, default: True
            Whether the axes aspect ratio follows the aspect ratio of the data
            limits.
        label_mode : {"L", "1", "all"}, default: "L"
            Determines which axes will get tick labels:

            - "L": All axes on the left column get vertical tick labels;
              all axes on the bottom row get horizontal tick labels.
            - "1": Only the bottom left axes is labelled.
            - "all": all axes are labelled.

        cbar_mode : {"each", "single", "edge", None}, default: None
            Whether to create a colorbar for "each" axes, a "single" colorbar
            for the entire grid, colorbars only for axes on the "edge"
            determined by *cbar_location*, or no colorbars.  The colorbars are
            stored in the :attr:`cbar_axes` attribute.
        cbar_location : {"left", "right", "bottom", "top"}, default: "right"
        cbar_pad : float, default: None
            Padding between the image axes and the colorbar axes.
        cbar_size : size specification (see `.Size.from_any`), default: "5%"
            Colorbar size.
        cbar_set_cax : bool, default: True
            If True, each axes in the grid has a *cax* attribute that is bound
            to associated *cbar_axes*.
        axes_class : subclass of `matplotlib.axes.Axes`, default: None
        
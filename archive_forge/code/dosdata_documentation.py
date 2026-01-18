from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
Simple 1-D plot of DOS data

        Data will be resampled onto a grid with `npts` points unless `npts` is
        set to zero, in which case:

        - no resampling takes place
        - `width` and `smearing` are ignored
        - `xmin` and `xmax` affect the axis limits of the plot, not the
          underlying data.

        If the special key 'label' is present in self.info, this will be set
        as the label for the plotted line (unless overruled in mplargs). The
        label is only seen if a legend is added to the plot (i.e. by calling
        ``ax.legend()``).

        Args:
            npts, xmin, xmax: output data range, as passed to self.sample_grid
            width: Width of broadening kernel, passed to self.sample_grid().
                If no npts was set but width is set, npts will be set to 1000.
            smearing: selection of broadening kernel for self.sample_grid()
            ax: existing Matplotlib axes object. If not provided, a new figure
                with one set of axes will be created using Pyplot
            show: show the figure on-screen
            filename: if a path is given, save the figure to this file
            mplargs: additional arguments to pass to matplotlib plot command
                (e.g. {'linewidth': 2} for a thicker line).

        Returns:
            Plotting axes. If "ax" was set, this is the same object.
        
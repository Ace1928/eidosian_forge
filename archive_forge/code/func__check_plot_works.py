from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
def _check_plot_works(f, default_axes=False, **kwargs):
    """
    Create plot and ensure that plot return object is valid.

    Parameters
    ----------
    f : func
        Plotting function.
    default_axes : bool, optional
        If False (default):
            - If `ax` not in `kwargs`, then create subplot(211) and plot there
            - Create new subplot(212) and plot there as well
            - Mind special corner case for bootstrap_plot (see `_gen_two_subplots`)
        If True:
            - Simply run plotting function with kwargs provided
            - All required axes instances will be created automatically
            - It is recommended to use it when the plotting function
            creates multiple axes itself. It helps avoid warnings like
            'UserWarning: To output multiple subplots,
            the figure containing the passed axes is being cleared'
    **kwargs
        Keyword arguments passed to the plotting function.

    Returns
    -------
    Plot object returned by the last plotting.
    """
    import matplotlib.pyplot as plt
    if default_axes:
        gen_plots = _gen_default_plot
    else:
        gen_plots = _gen_two_subplots
    ret = None
    try:
        fig = kwargs.get('figure', plt.gcf())
        plt.clf()
        for ret in gen_plots(f, fig, **kwargs):
            tm.assert_is_valid_plot_return_object(ret)
    finally:
        plt.close(fig)
    return ret
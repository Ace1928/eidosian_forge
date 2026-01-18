import numpy as np
plot the mean function and optionally the scatter of the sample

        Parameters
        ----------
        scatter : bool
            If true, then add scatterpoints of sample to plot.
        ax : None or matplotlib axis instance
            If None, then a matplotlib.pyplot figure is created, otherwise
            the given axis, ax, is used.

        Returns
        -------
        Figure
            This is either the created figure instance or the one associated
            with ax if ax is given.

        
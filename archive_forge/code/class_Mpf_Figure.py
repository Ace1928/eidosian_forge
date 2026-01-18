import matplotlib.pyplot as plt
import matplotlib.figure as mplfigure
import matplotlib.axes   as mpl_axes
from   mplfinance import _styles
import numpy as np
class Mpf_Figure(mplfigure.Figure):

    def add_subplot(self, *args, **kwargs):
        if 'style' in kwargs or not hasattr(self, 'mpfstyle'):
            style = _check_for_and_apply_style(kwargs)
        else:
            style = _check_for_and_apply_style(dict(style=self.mpfstyle))
        ax = mplfigure.Figure.add_subplot(self, *args, **kwargs)
        ax.mpfstyle = style
        return ax

    def add_axes(self, *args, **kwargs):
        if 'style' in kwargs or not hasattr(self, 'mpfstyle'):
            style = _check_for_and_apply_style(kwargs)
        else:
            style = _check_for_and_apply_style(dict(style=self.mpfstyle))
        ax = mplfigure.Figure.add_axes(self, *args, **kwargs)
        ax.mpfstyle = style
        return ax

    def subplot(self, *args, **kwargs):
        plt.figure(self.number)
        if 'style' in kwargs or not hasattr(self, 'mpfstyle'):
            style = _check_for_and_apply_style(kwargs)
        else:
            style = _check_for_and_apply_style(dict(style=self.mpfstyle))
        ax = plt.subplot(*args, **kwargs)
        ax.mpfstyle = style
        return ax

    def subplots(self, *args, **kwargs):
        if 'style' in kwargs or not hasattr(self, 'mpfstyle'):
            style = _check_for_and_apply_style(kwargs)
            self.mpfstyle = style
        else:
            style = _check_for_and_apply_style(dict(style=self.mpfstyle))
        axlist = mplfigure.Figure.subplots(self, *args, **kwargs)
        if isinstance(axlist, mpl_axes.Axes):
            axlist.mpfstyle = style
        elif isinstance(axlist, np.ndarray):
            for ax in axlist.flatten():
                ax.mpfstyle = style
        else:
            raise TypeError('Unexpected type (' + str(type(axlist)) + ') ' + 'returned from "matplotlib.figure.Figure.subplots()"')
        return axlist
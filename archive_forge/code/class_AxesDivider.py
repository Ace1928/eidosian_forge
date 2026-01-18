import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
class AxesDivider(Divider):
    """
    Divider based on the preexisting axes.
    """

    def __init__(self, axes, xref=None, yref=None):
        """
        Parameters
        ----------
        axes : :class:`~matplotlib.axes.Axes`
        xref
        yref
        """
        self._axes = axes
        if xref is None:
            self._xref = Size.AxesX(axes)
        else:
            self._xref = xref
        if yref is None:
            self._yref = Size.AxesY(axes)
        else:
            self._yref = yref
        super().__init__(fig=axes.get_figure(), pos=None, horizontal=[self._xref], vertical=[self._yref], aspect=None, anchor='C')

    def _get_new_axes(self, *, axes_class=None, **kwargs):
        axes = self._axes
        if axes_class is None:
            axes_class = type(axes)
        return axes_class(axes.get_figure(), axes.get_position(original=True), **kwargs)

    def new_horizontal(self, size, pad=None, pack_start=False, **kwargs):
        """
        Helper method for ``append_axes("left")`` and ``append_axes("right")``.

        See the documentation of `append_axes` for more details.

        :meta private:
        """
        if pad is None:
            pad = mpl.rcParams['figure.subplot.wspace'] * self._xref
        pos = 'left' if pack_start else 'right'
        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad, fraction_ref=self._xref)
            self.append_size(pos, pad)
        if not isinstance(size, Size._Base):
            size = Size.from_any(size, fraction_ref=self._xref)
        self.append_size(pos, size)
        locator = self.new_locator(nx=0 if pack_start else len(self._horizontal) - 1, ny=self._yrefindex)
        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)
        return ax

    def new_vertical(self, size, pad=None, pack_start=False, **kwargs):
        """
        Helper method for ``append_axes("top")`` and ``append_axes("bottom")``.

        See the documentation of `append_axes` for more details.

        :meta private:
        """
        if pad is None:
            pad = mpl.rcParams['figure.subplot.hspace'] * self._yref
        pos = 'bottom' if pack_start else 'top'
        if pad:
            if not isinstance(pad, Size._Base):
                pad = Size.from_any(pad, fraction_ref=self._yref)
            self.append_size(pos, pad)
        if not isinstance(size, Size._Base):
            size = Size.from_any(size, fraction_ref=self._yref)
        self.append_size(pos, size)
        locator = self.new_locator(nx=self._xrefindex, ny=0 if pack_start else len(self._vertical) - 1)
        ax = self._get_new_axes(**kwargs)
        ax.set_axes_locator(locator)
        return ax

    def append_axes(self, position, size, pad=None, *, axes_class=None, **kwargs):
        """
        Add a new axes on a given side of the main axes.

        Parameters
        ----------
        position : {"left", "right", "bottom", "top"}
            Where the new axes is positioned relative to the main axes.
        size : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            The axes width or height.  float or str arguments are interpreted
            as ``axes_size.from_any(size, AxesX(<main_axes>))`` for left or
            right axes, and likewise with ``AxesY`` for bottom or top axes.
        pad : :mod:`~mpl_toolkits.axes_grid1.axes_size` or float or str
            Padding between the axes.  float or str arguments are interpreted
            as for *size*.  Defaults to :rc:`figure.subplot.wspace` times the
            main Axes width (left or right axes) or :rc:`figure.subplot.hspace`
            times the main Axes height (bottom or top axes).
        axes_class : subclass type of `~.axes.Axes`, optional
            The type of the new axes.  Defaults to the type of the main axes.
        **kwargs
            All extra keywords arguments are passed to the created axes.
        """
        create_axes, pack_start = _api.check_getitem({'left': (self.new_horizontal, True), 'right': (self.new_horizontal, False), 'bottom': (self.new_vertical, True), 'top': (self.new_vertical, False)}, position=position)
        ax = create_axes(size, pad, pack_start=pack_start, axes_class=axes_class, **kwargs)
        self._fig.add_axes(ax)
        return ax

    def get_aspect(self):
        if self._aspect is None:
            aspect = self._axes.get_aspect()
            if aspect == 'auto':
                return False
            else:
                return True
        else:
            return self._aspect

    def get_position(self):
        if self._pos is None:
            bbox = self._axes.get_position(original=True)
            return bbox.bounds
        else:
            return self._pos

    def get_anchor(self):
        if self._anchor is None:
            return self._axes.get_anchor()
        else:
            return self._anchor

    def get_subplotspec(self):
        return self._axes.get_subplotspec()
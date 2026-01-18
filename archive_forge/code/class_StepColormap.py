import json
import math
import os
from jinja2 import Template
from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler
class StepColormap(ColorMap):
    """Creates a ColorMap based on linear interpolation of a set of colors
    over a given index.

    Parameters
    ----------
    colors : list-like object
        The set of colors to be used for interpolation.
        Colors can be provided in the form:
        * tuples of int between 0 and 255 (e.g: `(255,255,0)` or
        `(255, 255, 0, 255)`)
        * tuples of floats between 0. and 1. (e.g: `(1.,1.,0.)` or
        `(1., 1., 0., 1.)`)
        * HTML-like string (e.g: `"#ffff00`)
        * a color name or shortcut (e.g: `"y"` or `"yellow"`)
    index : list of floats, default None
        The bounds of the colors. The lower value is inclusive,
        the upper value is exclusive.
        It has to be sorted, and have the same length as `colors`.
        If None, a regular grid between `vmin` and `vmax` is created.
    vmin : float, default 0.
        The minimal value for the colormap.
        Values lower than `vmin` will be bound directly to `colors[0]`.
    vmax : float, default 1.
        The maximal value for the colormap.
        Values higher than `vmax` will be bound directly to `colors[-1]`.
    max_labels : int, default 10
        Maximum number of legend tick labels
    tick_labels: list of floats, default None
        If given, used as the positions of ticks.
    """

    def __init__(self, colors, index=None, vmin=0.0, vmax=1.0, caption='', max_labels=10, tick_labels=None):
        super().__init__(vmin=vmin, vmax=vmax, caption=caption, max_labels=max_labels)
        self.tick_labels = tick_labels
        n = len(colors)
        if n < 1:
            raise ValueError('You must provide at least 1 colors.')
        if index is None:
            self.index = [vmin + (vmax - vmin) * i * 1.0 / n for i in range(n + 1)]
        else:
            self.index = list(index)
        self.colors = [_parse_color(x) for x in colors]

    def rgba_floats_tuple(self, x):
        """
        Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with float values between 0. and 1.

        """
        if x <= self.index[0]:
            return self.colors[0]
        if x >= self.index[-1]:
            return self.colors[-1]
        i = len([u for u in self.index if u <= x])
        return tuple(self.colors[i - 1])

    def to_linear(self, index=None, max_labels=10):
        """
        Transforms the StepColormap into a LinearColormap.

        Parameters
        ----------
        index : list of floats, default None
                The values corresponding to each color in the output colormap.
                It has to be sorted.
                If None, a regular grid between `vmin` and `vmax` is created.
        max_labels : int, default 10
            Maximum number of legend tick labels

        """
        if index is None:
            n = len(self.index) - 1
            index = [self.index[i] * (1.0 - i / (n - 1.0)) + self.index[i + 1] * i / (n - 1.0) for i in range(n)]
        colors = [self.rgba_floats_tuple(x) for x in index]
        return LinearColormap(colors, index=index, vmin=self.vmin, vmax=self.vmax, max_labels=max_labels)

    def scale(self, vmin=0.0, vmax=1.0, max_labels=10):
        """Transforms the colorscale so that the minimal and maximal values
        fit the given parameters.
        """
        return StepColormap(self.colors, index=[vmin + (vmax - vmin) * (x - self.vmin) * 1.0 / (self.vmax - self.vmin) for x in self.index], vmin=vmin, vmax=vmax, caption=self.caption, max_labels=max_labels)
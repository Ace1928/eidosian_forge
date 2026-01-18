import json
import math
import os
from jinja2 import Template
from branca.element import ENV, Figure, JavascriptLink, MacroElement
from branca.utilities import legend_scaler
class LinearColormap(ColorMap):
    """Creates a ColorMap based on linear interpolation of a set of colors
    over a given index.

    Parameters
    ----------

    colors : list-like object with at least two colors.
        The set of colors to be used for interpolation.
        Colors can be provided in the form:
        * tuples of RGBA ints between 0 and 255 (e.g: `(255, 255, 0)` or
        `(255, 255, 0, 255)`)
        * tuples of RGBA floats between 0. and 1. (e.g: `(1.,1.,0.)` or
        `(1., 1., 0., 1.)`)
        * HTML-like string (e.g: `"#ffff00`)
        * a color name or shortcut (e.g: `"y"` or `"yellow"`)
    index : list of floats, default None
        The values corresponding to each color.
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
        If given, used as the positions of ticks."""

    def __init__(self, colors, index=None, vmin=0.0, vmax=1.0, caption='', max_labels=10, tick_labels=None):
        super().__init__(vmin=vmin, vmax=vmax, caption=caption, max_labels=max_labels)
        self.tick_labels = tick_labels
        n = len(colors)
        if n < 2:
            raise ValueError('You must provide at least 2 colors.')
        if index is None:
            self.index = [vmin + (vmax - vmin) * i * 1.0 / (n - 1) for i in range(n)]
        else:
            self.index = list(index)
        self.colors = [_parse_color(x) for x in colors]

    def rgba_floats_tuple(self, x):
        """Provides the color corresponding to value `x` in the
        form of a tuple (R,G,B,A) with float values between 0. and 1.
        """
        if x <= self.index[0]:
            return self.colors[0]
        if x >= self.index[-1]:
            return self.colors[-1]
        i = len([u for u in self.index if u < x])
        if self.index[i - 1] < self.index[i]:
            p = (x - self.index[i - 1]) * 1.0 / (self.index[i] - self.index[i - 1])
        elif self.index[i - 1] == self.index[i]:
            p = 1.0
        else:
            raise ValueError('Thresholds are not sorted.')
        return tuple(((1.0 - p) * self.colors[i - 1][j] + p * self.colors[i][j] for j in range(4)))

    def to_step(self, n=None, index=None, data=None, method=None, quantiles=None, round_method=None, max_labels=10):
        """Splits the LinearColormap into a StepColormap.

        Parameters
        ----------
        n : int, default None
            The number of expected colors in the output StepColormap.
            This will be ignored if `index` is provided.
        index : list of floats, default None
            The values corresponding to each color bounds.
            It has to be sorted.
            If None, a regular grid between `vmin` and `vmax` is created.
        data : list of floats, default None
            A sample of data to adapt the color map to.
        method : str, default 'linear'
            The method used to create data-based colormap.
            It can be 'linear' for linear scale, 'log' for logarithmic,
            or 'quant' for data's quantile-based scale.
        quantiles : list of floats, default None
            Alternatively, you can provide explicitly the quantiles you
            want to use in the scale.
        round_method : str, default None
            The method used to round thresholds.
            * If 'int', all values will be rounded to the nearest integer.
            * If 'log10', all values will be rounded to the nearest
            order-of-magnitude integer. For example, 2100 is rounded to
            2000, 2790 to 3000.
        max_labels : int, default 10
            Maximum number of legend tick labels

        Returns
        -------
        A StepColormap with `n=len(index)-1` colors.

        Examples:
        >> lc.to_step(n=12)
        >> lc.to_step(index=[0, 2, 4, 6, 8, 10])
        >> lc.to_step(data=some_list, n=12)
        >> lc.to_step(data=some_list, n=12, method='linear')
        >> lc.to_step(data=some_list, n=12, method='log')
        >> lc.to_step(data=some_list, n=12, method='quantiles')
        >> lc.to_step(data=some_list, quantiles=[0, 0.3, 0.7, 1])
        >> lc.to_step(data=some_list, quantiles=[0, 0.3, 0.7, 1],
        ...           round_method='log10')

        """
        msg = 'You must specify either `index` or `n`'
        if index is None:
            if data is None:
                if n is None:
                    raise ValueError(msg)
                else:
                    index = [self.vmin + (self.vmax - self.vmin) * i * 1.0 / n for i in range(1 + n)]
                    scaled_cm = self
            else:
                max_ = max(data)
                min_ = min(data)
                scaled_cm = self.scale(vmin=min_, vmax=max_)
                method = 'quantiles' if quantiles is not None else method if method is not None else 'linear'
                if method.lower().startswith('lin'):
                    if n is None:
                        raise ValueError(msg)
                    index = [min_ + i * (max_ - min_) * 1.0 / n for i in range(1 + n)]
                elif method.lower().startswith('log'):
                    if n is None:
                        raise ValueError(msg)
                    if min_ <= 0:
                        msg = 'Log-scale works only with strictly positive values.'
                        raise ValueError(msg)
                    index = [math.exp(math.log(min_) + i * (math.log(max_) - math.log(min_)) * 1.0 / n) for i in range(1 + n)]
                elif method.lower().startswith('quant'):
                    if quantiles is None:
                        if n is None:
                            msg = 'You must specify either `index`, `n` or`quantiles`.'
                            raise ValueError(msg)
                        else:
                            quantiles = [i * 1.0 / n for i in range(1 + n)]
                    p = len(data) - 1
                    s = sorted(data)
                    index = [s[int(q * p)] * (1.0 - q * p % 1) + s[min(int(q * p) + 1, p)] * (q * p % 1) for q in quantiles]
                else:
                    raise ValueError(f'Unknown method {method}')
        else:
            scaled_cm = self.scale(vmin=min(index), vmax=max(index))
        n = len(index) - 1
        if round_method == 'int':
            index = [round(x) for x in index]
        if round_method == 'log10':
            index = [_base(x) for x in index]
        colors = [scaled_cm.rgba_floats_tuple(index[i] * (1.0 - i / (n - 1.0)) + index[i + 1] * i / (n - 1.0)) for i in range(n)]
        caption = self.caption
        return StepColormap(colors, index=index, vmin=index[0], vmax=index[-1], caption=caption, max_labels=max_labels, tick_labels=self.tick_labels)

    def scale(self, vmin=0.0, vmax=1.0, max_labels=10):
        """Transforms the colorscale so that the minimal and maximal values
        fit the given parameters.
        """
        return LinearColormap(self.colors, index=[vmin + (vmax - vmin) * (x - self.vmin) * 1.0 / (self.vmax - self.vmin) for x in self.index], vmin=vmin, vmax=vmax, caption=self.caption, max_labels=max_labels)
import sys
from contextlib import suppress
import pandas as pd
from .._utils import array_kind
from ..exceptions import PlotnineError
from ..geoms import geom_blank
from ..mapping.aes import ALL_AESTHETICS, aes
from ..scales.scales import make_scale
class _lim:
    aesthetic = None

    def __init__(self, *limits):
        if not limits:
            msg = '{}lim(), is missing limits'
            raise PlotnineError(msg.format(self.aesthetic))
        elif len(limits) == 1:
            limits = limits[0]
        series = pd.Series(limits)
        if not any((x is None for x in limits)) and limits[0] > limits[1]:
            self.trans = 'reverse'
        elif array_kind.continuous(series):
            self.trans = 'identity'
        elif array_kind.discrete(series):
            self.trans = None
        elif array_kind.datetime(series):
            self.trans = 'datetime'
        elif array_kind.timedelta(series):
            self.trans = 'timedelta'
        else:
            msg = f'Unknown type {type(limits[0])} of limits'
            raise TypeError(msg)
        self.limits = limits
        self.limits_series = series

    def get_scale(self, plot):
        """
        Create a scale
        """
        ae = self.aesthetic
        series = self.limits_series
        ae_values = []
        for layer in plot.layers:
            with suppress(KeyError):
                value = layer.mapping[ae]
                if isinstance(value, str):
                    ae_values.append(value)
        for value in ae_values:
            if 'factor(' in value or 'Categorical(' in value:
                series = pd.Categorical(self.limits_series)
                break
        return make_scale(self.aesthetic, series, limits=self.limits, trans=self.trans)

    def __radd__(self, plot):
        scale = self.get_scale(plot)
        plot.scales.append(scale)
        return plot
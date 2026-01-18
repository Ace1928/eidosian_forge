from __future__ import annotations
import logging # isort:skip
from ..core.enums import Align, LabelOrientation
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_mixins import ScalarFillProps, ScalarLineProps, ScalarTextProps
from .formatters import (
from .labeling import AllLabels, LabelingPolicy
from .renderers import GuideRenderer
from .tickers import (
class MercatorAxis(LinearAxis):
    """ An axis that picks nice numbers for tick locations on a
    Mercator scale. Configured with a ``MercatorTickFormatter`` by default.

    Args:
        dimension ('lat' or 'lon', optional) :
            Whether this axis will display latitude or longitude values.
            (default: 'lat')

    """

    def __init__(self, dimension='lat', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(self.ticker, MercatorTicker):
            self.ticker.dimension = dimension
        if isinstance(self.formatter, MercatorTickFormatter):
            self.formatter.dimension = dimension
    ticker = Override(default=InstanceDefault(MercatorTicker))
    formatter = Override(default=InstanceDefault(MercatorTickFormatter))
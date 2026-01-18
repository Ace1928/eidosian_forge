import sys
from contextlib import suppress
import pandas as pd
from .._utils import array_kind
from ..exceptions import PlotnineError
from ..geoms import geom_blank
from ..mapping.aes import ALL_AESTHETICS, aes
from ..scales.scales import make_scale
class lims:
    """
    Set aesthetic limits

    Parameters
    ----------
    kwargs :
        Aesthetic and the values of the limits.
        e.g `x=(40, 100)`

    Notes
    -----
    If the 2nd value of `limits` is less than
    the first, a reversed scale will be created.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __radd__(self, plot):
        """
        Add limits to ggplot object
        """
        thismodule = sys.modules[__name__]
        for ae, value in self._kwargs.items():
            try:
                klass = getattr(thismodule, f'{ae}lim')
            except AttributeError as e:
                msg = "Cannot change limits for '{}'"
                raise PlotnineError(msg) from e
            plot += klass(value)
        return plot
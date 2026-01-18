from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_color_gradientn(scale_continuous):
    """
    Create a n color gradient

    Parameters
    ----------
    colors : list
        list of colors
    values : list, default=None
        list of points in the range [0, 1] at which to
        place each color. Must be the same size as
        `colors`. Default to evenly space the colors
    {superclass_parameters}
    na_value : str, default="#7F7F7F"
        Color of missing values

    See Also
    --------
    plotnine.scale_color_gradient
    plotnine.scale_color_gradientn
    """
    _aesthetics = ['color']
    guide = 'colorbar'
    na_value = '#7F7F7F'

    def __init__(self, colors, values=None, **kwargs):
        from mizani.palettes import gradient_n_pal
        self.palette = gradient_n_pal(colors, values)
        scale_continuous.__init__(self, **kwargs)
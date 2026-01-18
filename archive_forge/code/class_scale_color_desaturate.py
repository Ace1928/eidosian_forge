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
class scale_color_desaturate(scale_continuous):
    """
    Create a desaturated color gradient

    Parameters
    ----------
    color : str, default="red"
        Color to desaturate
    prop : float, default=0
        Saturation channel of color will be multiplied by
        this value.
    reverse : bool, default=False
        Whether to go from color to desaturated color
        or desaturated color to color.
    {superclass_parameters}
    na_value : str, default="#7F7F7F"
        Color of missing values.
    """
    _aesthetics = ['color']
    guide = 'colorbar'
    na_value = '#7F7F7F'

    def __init__(self, color='red', prop=0, reverse=False, **kwargs):
        from mizani.palettes import desaturate_pal
        self.palette = desaturate_pal(color, prop, reverse)
        scale_continuous.__init__(self, **kwargs)
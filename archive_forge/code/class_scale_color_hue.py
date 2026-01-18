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
class scale_color_hue(scale_discrete):
    """
    Qualitative color scale with evenly spaced hues

    Parameters
    ----------
    h :
        first hue. Must be in the range [0, 1]
    l :
        lightness. Must be in the range [0, 1]
    s :
        saturation. Must be in the range [0, 1]
    colorspace :
        Color space to use. Should be one of
        [hls](https://en.wikipedia.org/wiki/HSL_and_HSV)
        or
        [husl](http://www.husl-colors.org/).
    {superclass_parameters}
    na_value : str, default="#7F7F7F"
        Color of missing values.
    """
    _aesthetics = ['color']
    na_value = '#7F7F7F'

    def __init__(self, h: float=0.01, l: float=0.6, s: float=0.65, color_space: Literal['hls', 'husl']='hls', **kwargs):
        from mizani.palettes import hue_pal
        self._palette = hue_pal(h, l, s, color_space=color_space)
        scale_discrete.__init__(self, **kwargs)
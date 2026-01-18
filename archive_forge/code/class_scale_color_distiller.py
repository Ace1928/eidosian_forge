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
class scale_color_distiller(scale_color_gradientn):
    """
    Sequential and diverging continuous color scales

    This is a convinience scale around
    [](`~plotnine.scales.scale_color_gradientn`) with colors from
    [colorbrewer.org](http://colorbrewer2.org). It smoothly
    interpolates 7 colors from a brewer palette to create a
    continuous palette.

    Parameters
    ----------
    type :
        Type of data. Sequential, diverging or qualitative
    palette :
         If a string, will use that named palette.
         If a number, will index into the list of palettes
         of appropriate type. Default is 1
    values :
        list of points in the range [0, 1] at which to
        place each color. Must be the same size as
        `colors`. Default to evenly space the colors
    direction :
        Sets the order of colors in the scale. If 1
        colors are as output by [](`~mizani.palettes.brewer_pal`).
        If -1, the order of colors is reversed.
    {superclass_parameters}
    na_value : str, default="#7F7F7F"
        Color of missing values.
    """
    _aesthetics = ['color']
    guide = 'colorbar'
    na_value = '#7F7F7F'

    def __init__(self, type: ColorScheme | ColorSchemeShort='seq', palette: int | str=1, values: Optional[Sequence[float]]=None, direction: Literal[1, -1]=-1, **kwargs):
        """
        Create colormap that will be used by the palette
        """
        from mizani.palettes import brewer_pal
        if type.lower() in ('qual', 'qualitative'):
            warn("Using a discrete color palette in a continuous scale.Consider using type = 'seq' or type = 'div' instead", PlotnineWarning)
        colors = brewer_pal(type, palette, direction=direction)(7)
        scale_color_gradientn.__init__(self, colors, values, **kwargs)
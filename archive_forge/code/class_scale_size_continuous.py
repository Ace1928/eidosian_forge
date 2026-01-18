from warnings import warn
import numpy as np
from mizani.bounds import rescale_max
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_size_continuous(scale_continuous):
    """
    Continuous area size scale

    Parameters
    ----------
    range :
        Minimum and maximum area of the plotting symbol.
        It must be of size 2.
    {superclass_parameters}
    """
    _aesthetics = ['size']

    def __init__(self, range: tuple[float, float]=(1, 6), **kwargs):
        from mizani.palettes import area_pal
        self.palette = area_pal(range)
        scale_continuous.__init__(self, **kwargs)
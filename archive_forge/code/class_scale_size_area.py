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
class scale_size_area(scale_continuous):
    """
    Continuous area size scale

    Parameters
    ----------
    max_size :
        Maximum size of the plotting symbol.
    {superclass_parameters}
    """
    _aesthetics = ['size']
    rescaler = staticmethod(rescale_max)

    def __init__(self, max_size: float=6, **kwargs):
        from mizani.palettes import abs_area
        self.palette = abs_area(max_size)
        scale_continuous.__init__(self, **kwargs)
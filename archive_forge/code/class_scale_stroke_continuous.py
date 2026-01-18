from warnings import warn
import numpy as np
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_stroke_continuous(scale_continuous):
    """
    Continuous Stroke Scale

    Parameters
    ----------
    range :
        Range ([Minimum, Maximum]) of output stroke values.
        Should be between 0 and 1.
    {superclass_parameters}
    """
    _aesthetics = ['stroke']

    def __init__(self, range: tuple[float, float]=(1, 6), **kwargs):
        from mizani.palettes import rescale_pal
        self.palette = rescale_pal(range)
        scale_continuous.__init__(self, **kwargs)
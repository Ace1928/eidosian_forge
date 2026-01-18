from warnings import warn
import numpy as np
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_alpha_datetime(scale_datetime):
    """
    Datetime Alpha Scale

    Parameters
    ----------
    range : tuple
        Range ([Minimum, Maximum]) of output alpha values.
        Should be between 0 and 1.
    {superclass_parameters}
    """
    _aesthetics = ['alpha']

    def __init__(self, range: tuple[float, float]=(0.1, 1), **kwargs):
        from mizani.palettes import rescale_pal
        self.palette = rescale_pal(range)
        scale_datetime.__init__(self, **kwargs)
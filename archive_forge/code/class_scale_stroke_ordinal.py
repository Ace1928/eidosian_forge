from warnings import warn
import numpy as np
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_stroke_ordinal(scale_discrete):
    """
    Discrete Stroke Scale

    Parameters
    ----------
    range :
        Range ([Minimum, Maximum]) of output stroke values.
        Should be between 0 and 1.
    {superclass_parameters}
    """
    _aesthetics = ['stroke']

    def __init__(self, range: tuple[float, float]=(1, 6), **kwargs):

        def palette(value: int):
            return np.linspace(range[0], range[1], value)
        self.palette = palette
        scale_discrete.__init__(self, **kwargs)
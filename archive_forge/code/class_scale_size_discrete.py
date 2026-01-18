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
class scale_size_discrete(scale_size_ordinal):
    """
    Discrete area size scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['size']

    def __init__(self, **kwargs):
        warn('Using size for a discrete variable is not advised.', PlotnineWarning)
        super().__init__(**kwargs)
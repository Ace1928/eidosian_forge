from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_linetype_ordinal(scale_linetype):
    """
    Scale for line patterns

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['linetype']

    def __init__(self, **kwargs):
        warn('Using linetype for an ordinal variable is not advised.', PlotnineWarning)
        super().__init__(**kwargs)
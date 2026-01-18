from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_linetype(scale_discrete):
    """
    Scale for line patterns

    Parameters
    ----------
    {superclass_parameters}

    Notes
    -----
    The available linetypes are
    `'solid', 'dashed', 'dashdot', 'dotted'`
    If you need more custom linetypes, use
    [](`~plotnine.scales.scale_linetype_manual`)
    """
    _aesthetics = ['linetype']

    def __init__(self, **kwargs):
        from mizani.palettes import manual_pal
        self._palette = manual_pal(LINETYPES)
        super().__init__(**kwargs)
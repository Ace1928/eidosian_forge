from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@document
class scale_shape(scale_discrete):
    """
    Scale for shapes

    Parameters
    ----------
    unfilled :
        If `True`, then all shapes will have no interiors
        that can be a filled.
    {superclass_parameters}
    """
    _aesthetics = ['shape']

    def __init__(self, unfilled: bool=False, **kwargs):
        from mizani.palettes import manual_pal
        _shapes = unfilled_shapes if unfilled else shapes
        self._palette = manual_pal(_shapes)
        scale_discrete.__init__(self, **kwargs)
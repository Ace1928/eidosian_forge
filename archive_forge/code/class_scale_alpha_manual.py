from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_discrete import scale_discrete
@document
class scale_alpha_manual(_scale_manual):
    """
    Custom discrete alpha scale

    Parameters
    ----------
    values : array_like | dict
        Alpha values (in the [0, 1] range) that make up
        the palette. The values will be matched with the
        `limits` of the scale or the `breaks` if provided.
        If it is a dict then it should map data values to alpha
        values.
    {superclass_parameters}
    """
    _aesthetics = ['alpha']
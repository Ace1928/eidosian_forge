from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_discrete import scale_discrete
@document
class _scale_manual(scale_discrete):
    """
    Abstract class for manual scales

    Parameters
    ----------
    {superclass_parameters}
    """

    def __init__(self, values, **kwargs):
        if 'breaks' in kwargs:
            from collections.abc import Sized
            breaks: ScaleBreaksRaw = kwargs['breaks']
            if isinstance(breaks, Sized) and len(breaks) == len(values):
                values = dict(zip(breaks, values))
        self._values = values
        scale_discrete.__init__(self, **kwargs)

    def palette(self, value):
        max_n = len(self._values)
        if value > max_n:
            msg = f'The palette of {self.__class__.__name__} can return a maximum of {max_n} values. {value} were requested from it.'
            warn(msg, PlotnineWarning)
        return self._values
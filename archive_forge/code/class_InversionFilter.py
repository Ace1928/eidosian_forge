from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class InversionFilter(Filter):
    """ Inverts indices resulting from another filter. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    operand = Required(Instance(Filter), help='\n    Indices produced by this filter will be inverted.\n    ')
from __future__ import annotations
import logging # isort:skip
from abc import abstractmethod
from typing import Any
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
@abstract
class Dimensional(Model):
    """ A base class for models defining units of measurement.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    ticks = Required(List(Float), help='\n    Preferred values to choose from in non-exact mode.\n    ')
    include = Nullable(List(String), default=None, help='\n    An optional subset of preferred units from the basis.\n    ')
    exclude = List(String, default=[], help='\n    A subset of units from the basis to avoid.\n    ')

    @abstractmethod
    def is_known(self, unit: str) -> bool:
        pass
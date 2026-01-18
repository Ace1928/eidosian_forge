from __future__ import annotations
import logging # isort:skip
from abc import abstractmethod
from typing import Any
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
class Angular(CustomDimensional):
    """ Units of angular measurement.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    basis = Override(default={'°': (1, '^\\circ', 'degree'), "'": (1 / 60, '^\\prime', 'minute'), "''": (1 / 3600, '^{\\prime\\prime}', 'second')})
    ticks = Override(default=[1, 3, 6, 12, 60, 120, 240, 360])
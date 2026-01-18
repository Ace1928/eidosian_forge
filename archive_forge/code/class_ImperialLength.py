from __future__ import annotations
import logging # isort:skip
from abc import abstractmethod
from typing import Any
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
class ImperialLength(CustomDimensional):
    """ Imperial units of length measurement.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    basis = Override(default={'in': (1 / 12, 'in', 'inch'), 'ft': (1, 'ft', 'foot'), 'yd': (3, 'yd', 'yard'), 'ch': (66, 'ch', 'chain'), 'fur': (660, 'fur', 'furlong'), 'mi': (5280, 'mi', 'mile'), 'lea': (15840, 'lea', 'league')})
    ticks = Override(default=[1, 3, 6, 12, 60])
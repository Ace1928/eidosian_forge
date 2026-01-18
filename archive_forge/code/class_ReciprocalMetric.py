from __future__ import annotations
import logging # isort:skip
from abc import abstractmethod
from typing import Any
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
class ReciprocalMetric(Metric):
    """ Model for defining reciprocal metric units of measurement, e.g. ``m^{-1}``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
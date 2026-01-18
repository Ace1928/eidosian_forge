from __future__ import annotations
import logging # isort:skip
from math import inf
from ..core.enums import Direction
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
@abstract
class ScalarExpression(Model):
    """ Base class for for scalar expressions. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
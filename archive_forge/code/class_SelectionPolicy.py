from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
@abstract
class SelectionPolicy(Model):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
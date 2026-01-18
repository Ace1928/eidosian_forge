from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class AllLabels(LabelingPolicy):
    """ Select all labels even if they overlap. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class AllIndices(Filter):
    """ Trivial filter that includes all indices in a dataset. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class IndexFilter(Filter):
    """ An ``IndexFilter`` filters data by returning the subset of data at a given set of indices.
    """
    indices = Nullable(Seq(Int), help='\n    A list of integer indices representing the subset of data to select.\n    ')

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and 'indices' not in kwargs:
            kwargs['indices'] = args[0]
        super().__init__(**kwargs)
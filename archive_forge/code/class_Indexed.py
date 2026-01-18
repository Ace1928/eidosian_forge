from __future__ import annotations
import logging # isort:skip
from typing import Any, ClassVar, Literal
from ..core.properties import (
from ..core.property.aliases import CoordinateLike
from ..model import Model
class Indexed(Coordinate):
    """ A coordinate computed given an index into a renderer's data.

    .. note::
        This model is experimental and may change at any point.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
    index = Required(Int, help='\n    An index into the data.\n    ')
    renderer = Instance('.models.renderers.GlyphRenderer', help='\n    A renderer that is the provider of the data.\n    ')
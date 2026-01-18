from __future__ import annotations
import logging # isort:skip
from .. import palettes
from ..core.enums import Palette
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error, warning
from ..core.validation.errors import WEIGHTED_STACK_COLOR_MAPPER_LABEL_LENGTH_MISMATCH
from ..core.validation.warnings import PALETTE_LENGTH_FACTORS_MISMATCH
from .transforms import Transform
class CategoricalMarkerMapper(CategoricalMapper):
    """ Map categorical factors to marker types.

    Values that are passed to this mapper that are not in the factors list
    will be mapped to ``default_value``.

    .. note::
        This mappers is primarily only useful with the ``Scatter`` marker
        glyph that be parameterized by marker type.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    markers = Seq(MarkerType, help='\n    A sequence of marker types to use as the target for mapping.\n    ')
    default_value = MarkerType(default='circle', help='\n    A marker type to use in case an unrecognized factor is passed in to be\n    mapped.\n    ')
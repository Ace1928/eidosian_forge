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
class EqHistColorMapper(ScanningColorMapper):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    bins = Int(default=256 * 256, help='Number of histogram bins')
    rescale_discrete_levels = Bool(default=False, help='\n    If there are only a few discrete levels in the values that are color\n    mapped then ``rescale_discrete_levels=True`` decreases the lower limit of\n    the span so that the values are rendered towards the top end of the\n    palette.\n    ')
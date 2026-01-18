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
class CategoricalColorMapper(CategoricalMapper, ColorMapper):
    """ Map categorical factors to colors.

    Values that are passed to this mapper that are not in the factors list
    will be mapped to ``nan_color``.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @warning(PALETTE_LENGTH_FACTORS_MISMATCH)
    def _check_palette_length(self):
        palette = self.palette
        factors = self.factors
        if len(palette) < len(factors):
            extra_factors = factors[len(palette):]
            return f'{extra_factors} will be assigned to `nan_color` {self.nan_color}'
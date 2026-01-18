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
@error(WEIGHTED_STACK_COLOR_MAPPER_LABEL_LENGTH_MISMATCH)
def _check_label_length(self):
    if self.stack_labels is not None:
        nlabel = len(self.stack_labels)
        npalette = len(self.palette)
        if nlabel > npalette:
            self.stack_labels = self.stack_labels[:npalette]
            return f'{nlabel} != {npalette}, removing unwanted stack_labels'
        elif nlabel < npalette:
            self.stack_labels = list(self.stack_labels) + [''] * (npalette - nlabel)
            return f'{nlabel} != {npalette}, padding with empty strings'
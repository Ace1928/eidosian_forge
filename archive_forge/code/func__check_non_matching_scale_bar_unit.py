from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.vectorization import Field
from ...core.property_mixins import (
from ...core.validation import error
from ...core.validation.errors import (
from ...model import Model
from ..formatters import TickFormatter
from ..labeling import LabelingPolicy, NoOverlap
from ..mappers import ColorMapper
from ..ranges import Range
from ..renderers import GlyphRenderer
from ..tickers import FixedTicker, Ticker
from .annotation import Annotation
from .dimensional import Dimensional, MetricLength
@error(NON_MATCHING_SCALE_BAR_UNIT)
def _check_non_matching_scale_bar_unit(self):
    if not self.dimensional.is_known(self.unit):
        return str(self)
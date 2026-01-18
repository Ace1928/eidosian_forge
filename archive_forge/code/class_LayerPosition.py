from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from xformers.components import NormalizationType, ResidualNormStyle
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, FeedforwardConfig
from xformers.components.positional_embedding import (
from xformers.utils import generate_matching_config
class LayerPosition:
    """Bitmask to mark this layer as first, last, nothing or both"""

    def __init__(self):
        self.bitmask = LayerPositionBitmask.Default

    def is_first(self):
        return bool(self.bitmask & LayerPositionBitmask.First)

    def is_last(self):
        return bool(self.bitmask & LayerPositionBitmask.Last)

    def mark_not_first(self):
        self.bitmask &= ~LayerPositionBitmask.First

    def mark_not_last(self):
        self.bitmask &= ~LayerPositionBitmask.Last
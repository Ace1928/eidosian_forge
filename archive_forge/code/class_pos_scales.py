from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class pos_scales:
    """
    Position Scales
    """
    x: scale
    y: scale
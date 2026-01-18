from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class panel_ranges:
    """
    Ranges for the panel
    """
    x: TupleFloat2
    y: TupleFloat2
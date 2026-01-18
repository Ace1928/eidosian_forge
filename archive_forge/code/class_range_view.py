from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class range_view:
    """
    Range information after trainning
    """
    range: TupleFloat2
    range_coord: TupleFloat2
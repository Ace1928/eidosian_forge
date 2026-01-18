from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class mpl_save_view:
    """
    Everything required to save a matplotlib figure
    """
    figure: Figure
    kwargs: dict[str, Any]
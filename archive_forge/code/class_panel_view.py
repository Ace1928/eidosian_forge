from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class panel_view:
    """
    Information from the trained position scales in a panel
    """
    x: scale_view
    y: scale_view
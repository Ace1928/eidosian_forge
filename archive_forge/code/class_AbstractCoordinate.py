from __future__ import annotations
from enum import Enum
from typing import NamedTuple, Union, Optional, NewType, Any, List
import numpy as np
from qiskit import pulse
class AbstractCoordinate(str, Enum):
    """Abstract coordinate that the exact value depends on the user preference.

    RIGHT: The horizontal coordinate at t0 shifted by the left margin.
    LEFT: The horizontal coordinate at tf shifted by the right margin.
    TOP: The vertical coordinate at the top of chart.
    BOTTOM: The vertical coordinate at the bottom of chart.
    """
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    TOP = 'TOP'
    BOTTOM = 'BOTTOM'
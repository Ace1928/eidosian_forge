from __future__ import annotations
from enum import Enum
from typing import NamedTuple, Union, Optional, NewType, Any, List
import numpy as np
from qiskit import pulse
class HorizontalAxis(NamedTuple):
    window: tuple[int, int]
    axis_map: dict[float, float | str]
    axis_break_pos: list[int]
    label: str
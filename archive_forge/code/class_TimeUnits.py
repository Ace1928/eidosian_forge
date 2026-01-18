from __future__ import annotations
from enum import Enum
from typing import NamedTuple, Union, Optional, NewType, Any, List
import numpy as np
from qiskit import pulse
class TimeUnits(str, Enum):
    """Representation of time units.

    SYSTEM_CYCLE_TIME: System time dt.
    NANO_SEC: Nano seconds.
    """
    CYCLES = 'dt'
    NS = 'ns'
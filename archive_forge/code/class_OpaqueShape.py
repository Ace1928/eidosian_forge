from __future__ import annotations
from enum import Enum
from typing import NamedTuple, Union, Optional, NewType, Any, List
import numpy as np
from qiskit import pulse
class OpaqueShape(NamedTuple):
    duration: np.ndarray
    meta: dict[str, Any]
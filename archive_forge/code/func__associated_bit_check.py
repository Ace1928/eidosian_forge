from __future__ import annotations
import warnings
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from enum import Enum
import numpy as np
from qiskit import circuit
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import drawings, types
from qiskit.visualization.timeline.stylesheet import QiskitTimelineStyle
def _associated_bit_check(_data):
    """If any associated bit is not shown."""
    if all((bit not in self.assigned_coordinates for bit in _data.bits)):
        return False
    return True
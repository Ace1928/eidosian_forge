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
def _data_check(_data):
    """If data is valid."""
    if _data.data_type == str(types.LineType.GATE_LINK.value):
        active_bits = [bit for bit in _data.bits if bit not in self.disable_bits]
        if len(active_bits) < 2:
            return False
    elif _data.data_type in _barriers and (not self.formatter['control.show_barriers']):
        return False
    elif _data.data_type in _delays and (not self.formatter['control.show_delays']):
        return False
    return True
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
def _time_range_check(_data):
    """If data is located outside the current time range."""
    t0, t1 = self.time_range
    if np.max(_data.xvals) < t0 or np.min(_data.xvals) > t1:
        return False
    return True
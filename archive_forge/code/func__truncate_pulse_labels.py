from __future__ import annotations
from collections.abc import Iterator, Sequence
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain
import numpy as np
from qiskit import pulse
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawings, device_info
from qiskit.visualization.pulse_v2.stylesheet import QiskitPulseStyle
def _truncate_pulse_labels(self, xvals: np.ndarray, yvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A helper function to remove text according to time breaks.

        Args:
            xvals: Time points.
            yvals: Data points.

        Returns:
            Set of truncated numpy arrays for x and y coordinate.
        """
    xpos = xvals[0]
    t0, t1 = self.parent.time_range
    if xpos < t0 or xpos > t1:
        return (np.array([]), np.array([]))
    offset_accumulation = 0
    for tl, tr in self.parent.time_breaks:
        if xpos < tl:
            return (np.array([xpos - offset_accumulation]), yvals)
        if tl < xpos < tr:
            return (np.array([tl - offset_accumulation]), yvals)
        else:
            offset_accumulation += tr - tl
    return (np.array([xpos - offset_accumulation]), yvals)
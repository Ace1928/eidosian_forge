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
def _truncate_boxes(self, xvals: np.ndarray, yvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A helper function to clip box object according to time breaks.

        Args:
            xvals: Time points.
            yvals: Data points.

        Returns:
            Set of truncated numpy arrays for x and y coordinate.
        """
    x0, x1 = xvals
    t0, t1 = self.parent.time_range
    if x1 < t0 or x0 > t1:
        return (np.array([]), np.array([]))
    x0 = max(t0, x0)
    x1 = min(t1, x1)
    offset_accumulate = 0
    for tl, tr in self.parent.time_breaks:
        tl -= offset_accumulate
        tr -= offset_accumulate
        if x1 < tl:
            break
        if tl < x0 and tr > x1:
            return (np.array([]), np.array([]))
        elif tl < x1 < tr:
            x1 = tl
        elif tl < x0 < tr:
            x0 = tl
            x1 = tl + t1 - tr
        elif tl > x0 and tr < x1:
            x1 -= tr - tl
        elif tr < x0:
            x0 -= tr - tl
            x1 -= tr - tl
        offset_accumulate += tr - tl
    return (np.asarray([x0, x1], dtype=float), yvals)
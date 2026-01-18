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
def _truncate_vectors(self, xvals: np.ndarray, yvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A helper function to remove sequential data points according to time breaks.

        Args:
            xvals: Time points.
            yvals: Data points.

        Returns:
            Set of truncated numpy arrays for x and y coordinate.
        """
    xvals = np.asarray(xvals, dtype=float)
    yvals = np.asarray(yvals, dtype=float)
    t0, t1 = self.parent.time_range
    if max(xvals) < t0 or min(xvals) > t1:
        return (np.array([]), np.array([]))
    if min(xvals) < t0:
        inds = xvals > t0
        yvals = np.append(np.interp(t0, xvals, yvals), yvals[inds])
        xvals = np.append(t0, xvals[inds])
    if max(xvals) > t1:
        inds = xvals < t1
        yvals = np.append(yvals[inds], np.interp(t1, xvals, yvals))
        xvals = np.append(xvals[inds], t1)
    trunc_xvals = [xvals]
    trunc_yvals = [yvals]
    offset_accumulate = 0
    for tl, tr in self.parent.time_breaks:
        sub_xs = trunc_xvals.pop()
        sub_ys = trunc_yvals.pop()
        tl -= offset_accumulate
        tr -= offset_accumulate
        min_xs = min(sub_xs)
        max_xs = max(sub_xs)
        if max_xs < tl:
            trunc_xvals.append(sub_xs)
            trunc_yvals.append(sub_ys)
            break
        if tl < min_xs and tr > max_xs:
            return (np.array([]), np.array([]))
        elif tl < max_xs < tr:
            inds = sub_xs > tl
            trunc_xvals.append(np.append(tl, sub_xs[inds]) - (tl - min_xs))
            trunc_yvals.append(np.append(np.interp(tl, sub_xs, sub_ys), sub_ys[inds]))
        elif tl < min_xs < tr:
            inds = sub_xs < tr
            trunc_xvals.append(np.append(sub_xs[inds], tr))
            trunc_yvals.append(np.append(sub_ys[inds], np.interp(tr, sub_xs, sub_ys)))
        elif tl > min_xs and tr < max_xs:
            inds0 = sub_xs < tl
            trunc_xvals.append(np.append(sub_xs[inds0], tl))
            trunc_yvals.append(np.append(sub_ys[inds0], np.interp(tl, sub_xs, sub_ys)))
            inds1 = sub_xs > tr
            trunc_xvals.append(np.append(tr, sub_xs[inds1]) - (tr - tl))
            trunc_yvals.append(np.append(np.interp(tr, sub_xs, sub_ys), sub_ys[inds1]))
        elif tr < min_xs:
            trunc_xvals.append(sub_xs - (tr - tl))
            trunc_yvals.append(sub_ys)
        else:
            trunc_xvals.append(sub_xs)
            trunc_yvals.append(sub_ys)
        offset_accumulate += tr - tl
    new_x = np.concatenate(trunc_xvals)
    new_y = np.concatenate(trunc_yvals)
    return (np.asarray(new_x, dtype=float), np.asarray(new_y, dtype=float))
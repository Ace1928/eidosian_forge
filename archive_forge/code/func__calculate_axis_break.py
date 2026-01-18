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
def _calculate_axis_break(self, program: pulse.Schedule) -> list[tuple[int, int]]:
    """A helper function to calculate axis break of long pulse sequence.

        Args:
            program: A schedule to calculate axis break.

        Returns:
            List of axis break periods.
        """
    axis_breaks = []
    edges = set()
    for t0, t1 in chain.from_iterable(program.timeslots.values()):
        if t1 - t0 > 0:
            edges.add(t0)
            edges.add(t1)
    edges = sorted(edges)
    for t0, t1 in zip(edges[:-1], edges[1:]):
        if t1 - t0 > self.formatter['axis_break.length']:
            t_l = t0 + 0.5 * self.formatter['axis_break.max_length']
            t_r = t1 - 0.5 * self.formatter['axis_break.max_length']
            axis_breaks.append((t_l, t_r))
    return axis_breaks
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
def _check_data_visible(self, data: drawings.ElementaryData) -> bool:
    """A helper function to check if the data is visible.

        Args:
            data: Drawing object to test.

        Returns:
            Return `True` if the data is visible.
        """
    _barriers = [str(types.LineType.BARRIER.value)]
    _delays = [str(types.BoxType.DELAY.value), str(types.LabelType.DELAY.value)]

    def _time_range_check(_data):
        """If data is located outside the current time range."""
        t0, t1 = self.time_range
        if np.max(_data.xvals) < t0 or np.min(_data.xvals) > t1:
            return False
        return True

    def _associated_bit_check(_data):
        """If any associated bit is not shown."""
        if all((bit not in self.assigned_coordinates for bit in _data.bits)):
            return False
        return True

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
    checks = [_time_range_check, _associated_bit_check, _data_check]
    if all((check(data) for check in checks)):
        return True
    return False
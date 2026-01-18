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
def _truncate_data(self, data: drawings.ElementaryData) -> tuple[np.ndarray, np.ndarray]:
    """A helper function to truncate drawings according to time breaks.

        # TODO: move this function to common module to support axis break for timeline.

        Args:
            data: Drawing object to truncate.

        Returns:
            Set of truncated numpy arrays for x and y coordinate.
        """
    xvals = self._bind_coordinate(data.xvals)
    yvals = self._bind_coordinate(data.yvals)
    if isinstance(data, drawings.BoxData):
        return self._truncate_boxes(xvals, yvals)
    elif data.data_type in [types.LabelType.PULSE_NAME, types.LabelType.OPAQUE_BOXTEXT]:
        return self._truncate_pulse_labels(xvals, yvals)
    else:
        return self._truncate_vectors(xvals, yvals)
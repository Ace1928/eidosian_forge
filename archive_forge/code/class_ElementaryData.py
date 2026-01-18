from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any
import numpy as np
from qiskit.pulse.channels import Channel
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.exceptions import VisualizationError
class ElementaryData(ABC):
    """Base class of the pulse visualization interface."""
    __hash__ = None

    def __init__(self, data_type: str | Enum, xvals: np.ndarray, yvals: np.ndarray, channels: Channel | list[Channel] | None=None, meta: dict[str, Any] | None=None, ignore_scaling: bool=False, styles: dict[str, Any] | None=None):
        """Create new drawing.

        Args:
            data_type: String representation of this drawing.
            xvals: Series of horizontal coordinate that the object is drawn.
            yvals: Series of vertical coordinate that the object is drawn.
            channels: Pulse channel object bound to this drawing.
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        if channels and isinstance(channels, Channel):
            channels = [channels]
        if isinstance(data_type, Enum):
            data_type = data_type.value
        self.data_type = str(data_type)
        self.xvals = np.array(xvals, dtype=object)
        self.yvals = np.array(yvals, dtype=object)
        self.channels: list[Channel] = channels or []
        self.meta = meta or {}
        self.ignore_scaling = ignore_scaling
        self.styles = styles or {}

    @property
    def data_key(self):
        """Return unique hash of this object."""
        return str(hash((self.__class__.__name__, self.data_type, tuple(self.xvals), tuple(self.yvals))))

    def __repr__(self):
        return f'{self.__class__.__name__}(type={self.data_type}, key={self.data_key})'

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.data_key == other.data_key
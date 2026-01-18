import uuid
from typing import Generic, TypeVar, Optional
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import MeasurementProcess, MidMeasure
def _transform_bin_op(self, base_bin, other):
    """Helper function for defining dunder binary operations."""
    if isinstance(other, MeasurementValue):
        return self._merge(other)._apply(lambda t: base_bin(t[0], t[1]))
    return self._apply(lambda v: base_bin(v, other))
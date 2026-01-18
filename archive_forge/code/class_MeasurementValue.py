import uuid
from typing import Generic, TypeVar, Optional
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import MeasurementProcess, MidMeasure
class MeasurementValue(Generic[T]):
    """A class representing unknown measurement outcomes in the qubit model.

    Measurements on a single qubit in the computational basis are assumed.

    Args:
        measurements (list[.MidMeasureMP]): The measurement(s) that this object depends on.
        processing_fn (callable): A lazily transformation applied to the measurement values.
    """
    name = 'MeasurementValue'

    def __init__(self, measurements, processing_fn):
        self.measurements = measurements
        self.processing_fn = processing_fn

    def _items(self):
        """A generator representing all the possible outcomes of the MeasurementValue."""
        for i in range(2 ** len(self.measurements)):
            branch = tuple((int(b) for b in np.binary_repr(i, width=len(self.measurements))))
            yield (branch, self.processing_fn(*branch))

    @property
    def wires(self):
        """Returns a list of wires corresponding to the mid-circuit measurements."""
        return Wires.all_wires([m.wires for m in self.measurements])

    @property
    def branches(self):
        """A dictionary representing all possible outcomes of the MeasurementValue."""
        ret_dict = {}
        for i in range(2 ** len(self.measurements)):
            branch = tuple((int(b) for b in np.binary_repr(i, width=len(self.measurements))))
            ret_dict[branch] = self.processing_fn(*branch)
        return ret_dict

    def map_wires(self, wire_map):
        """Returns a copy of the current ``MeasurementValue`` with the wires of each measurement changed
        according to the given wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            MeasurementValue: new ``MeasurementValue`` instance with measurement wires mapped
        """
        mapped_measurements = [m.map_wires(wire_map) for m in self.measurements]
        return MeasurementValue(mapped_measurements, self.processing_fn)

    def _transform_bin_op(self, base_bin, other):
        """Helper function for defining dunder binary operations."""
        if isinstance(other, MeasurementValue):
            return self._merge(other)._apply(lambda t: base_bin(t[0], t[1]))
        return self._apply(lambda v: base_bin(v, other))

    def __invert__(self):
        """Return a copy of the measurement value with an inverted control
        value."""
        return self._apply(lambda v: not v)

    def __eq__(self, other):
        return self._transform_bin_op(lambda a, b: a == b, other)

    def __ne__(self, other):
        return self._transform_bin_op(lambda a, b: a != b, other)

    def __add__(self, other):
        return self._transform_bin_op(lambda a, b: a + b, other)

    def __radd__(self, other):
        return self._apply(lambda v: other + v)

    def __sub__(self, other):
        return self._transform_bin_op(lambda a, b: a - b, other)

    def __rsub__(self, other):
        return self._apply(lambda v: other - v)

    def __mul__(self, other):
        return self._transform_bin_op(lambda a, b: a * b, other)

    def __rmul__(self, other):
        return self._apply(lambda v: other * v)

    def __truediv__(self, other):
        return self._transform_bin_op(lambda a, b: a / b, other)

    def __rtruediv__(self, other):
        return self._apply(lambda v: other / v)

    def __lt__(self, other):
        return self._transform_bin_op(lambda a, b: a < b, other)

    def __le__(self, other):
        return self._transform_bin_op(lambda a, b: a <= b, other)

    def __gt__(self, other):
        return self._transform_bin_op(lambda a, b: a > b, other)

    def __ge__(self, other):
        return self._transform_bin_op(lambda a, b: a >= b, other)

    def __and__(self, other):
        return self._transform_bin_op(lambda a, b: a and b, other)

    def __or__(self, other):
        return self._transform_bin_op(lambda a, b: a or b, other)

    def _apply(self, fn):
        """Apply a post computation to this measurement"""
        return MeasurementValue(self.measurements, lambda *x: fn(self.processing_fn(*x)))

    def concretize(self, measurements: dict):
        """Returns a concrete value from a dictionary of hashes with concrete values."""
        values = tuple((measurements[meas] for meas in self.measurements))
        return self.processing_fn(*values)

    def _merge(self, other: 'MeasurementValue'):
        """Merge two measurement values"""
        merged_measurements = list(set(self.measurements).union(set(other.measurements)))
        merged_measurements.sort(key=lambda m: m.id)

        def merged_fn(*x):
            sub_args_1 = (x[i] for i in [merged_measurements.index(m) for m in self.measurements])
            sub_args_2 = (x[i] for i in [merged_measurements.index(m) for m in other.measurements])
            out_1 = self.processing_fn(*sub_args_1)
            out_2 = other.processing_fn(*sub_args_2)
            return (out_1, out_2)
        return MeasurementValue(merged_measurements, merged_fn)

    def __getitem__(self, i):
        branch = tuple((int(b) for b in np.binary_repr(i, width=len(self.measurements))))
        return self.processing_fn(*branch)

    def __str__(self):
        lines = []
        for i in range(2 ** len(self.measurements)):
            branch = tuple((int(b) for b in np.binary_repr(i, width=len(self.measurements))))
            id_branch_mapping = [f'{self.measurements[j].id}={branch[j]}' for j in range(len(branch))]
            lines.append('if ' + ','.join(id_branch_mapping) + ' => ' + str(self.processing_fn(*branch)))
        return '\n'.join(lines)

    def __repr__(self):
        return f'MeasurementValue(wires={self.wires.tolist()})'
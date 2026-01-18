import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
@value.value_equality
class FixedDurationUndirectedGraphDeviceEdge(UndirectedGraphDeviceEdge):
    """An edge of an undirected graph device on which every operation is
    allowed and has the same duration."""

    def __init__(self, duration: value.Duration) -> None:
        self._duration = duration

    def duration_of(self, operation: ops.Operation) -> value.Duration:
        return self._duration

    def validate_operation(self, operation: ops.Operation) -> None:
        pass

    def _value_equality_values_(self):
        return self._duration
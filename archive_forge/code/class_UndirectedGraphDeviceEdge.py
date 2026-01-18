import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
class UndirectedGraphDeviceEdge(metaclass=abc.ABCMeta):
    """An edge of an undirected graph device."""

    @abc.abstractmethod
    def duration_of(self, operation: ops.Operation) -> value.Duration:
        pass

    @abc.abstractmethod
    def validate_operation(self, operation: ops.Operation) -> None:
        pass
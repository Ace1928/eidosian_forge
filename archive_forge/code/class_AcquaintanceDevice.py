from typing import Union, TYPE_CHECKING
import abc
from cirq import circuits, devices, ops
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate, SwapNetworkGate
from cirq.contrib.acquaintance.bipartite import BipartiteSwapNetworkGate
from cirq.contrib.acquaintance.shift_swap_network import ShiftSwapNetworkGate
from cirq.contrib.acquaintance.permutation import PermutationGate
class AcquaintanceDevice(devices.Device, metaclass=abc.ABCMeta):
    """A device that contains only acquaintance and permutation gates."""
    gate_types = (AcquaintanceOpportunityGate, PermutationGate)

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        if not (isinstance(operation, ops.GateOperation) and isinstance(operation.gate, self.gate_types)):
            raise ValueError(f'not (isinstance({operation!r}, {ops.Operation!r}) and ininstance({operation!r}.gate, {self.gate_types!r})')
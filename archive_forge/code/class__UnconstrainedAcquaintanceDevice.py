from typing import Union, TYPE_CHECKING
import abc
from cirq import circuits, devices, ops
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate, SwapNetworkGate
from cirq.contrib.acquaintance.bipartite import BipartiteSwapNetworkGate
from cirq.contrib.acquaintance.shift_swap_network import ShiftSwapNetworkGate
from cirq.contrib.acquaintance.permutation import PermutationGate
class _UnconstrainedAcquaintanceDevice(AcquaintanceDevice):
    """An acquaintance device with no constraints other than of the gate types."""

    def __repr__(self) -> str:
        return 'UnconstrainedAcquaintanceDevice'
import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
class DecomposePermutationGates:

    def __init__(self, keep_swap_permutations: bool=True):
        """Decomposes permutation gates.

        Args:
            keep_swap_permutations: Whether or not to except
                SwapPermutationGate.
        """
        if keep_swap_permutations:
            self.no_decomp = lambda op: not all([isinstance(op, ops.GateOperation), isinstance(op.gate, PermutationGate), not isinstance(op.gate, SwapPermutationGate)])
        else:
            self.no_decomp = lambda op: not all([isinstance(op, ops.GateOperation), isinstance(op.gate, PermutationGate)])

    def optimize_circuit(self, circuit: 'cirq.Circuit') -> None:
        circuit._moments = [*transformers.expand_composite(circuit, no_decomp=self.no_decomp)]

    def __call__(self, circuit: 'cirq.Circuit') -> None:
        self.optimize_circuit(circuit)
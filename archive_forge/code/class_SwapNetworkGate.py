import functools
import itertools
import math
import operator
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, protocols, value
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
@value.value_equality
class SwapNetworkGate(PermutationGate):
    """A single gate representing a generalized swap network.

    Args:
        part_lens: An sequence indicating the sizes of the parts in the
            partition defining the swap network.
        acquaintance_size: An int indicating the locality of the logical gates
            desired; used to keep track of this while nesting. If 0, no
            acquaintance gates are inserted. If None, after each pair of parts
            is shifted the union thereof is acquainted.
        swap_gate: The gate used to swap logical indices.

    Attributes:
        part_lens: See above.
        acquaintance_size: See above.
        swap_gate: The gate used to swap logical indices.
    """

    def __init__(self, part_lens: Sequence[int], acquaintance_size: Optional[int]=0, swap_gate: 'cirq.Gate'=ops.SWAP) -> None:
        super().__init__(sum(part_lens), swap_gate)
        if len(part_lens) < 2:
            raise ValueError('len(part_lens) < 2.')
        self.part_lens = tuple(part_lens)
        self.acquaintance_size = acquaintance_size

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        qubit_to_position = {q: i for i, q in enumerate(qubits)}
        mapping = dict(qubit_to_position)
        parts = []
        n_qubits = 0
        for part_len in self.part_lens:
            parts.append(list(qubits[n_qubits:n_qubits + part_len]))
            n_qubits += part_len
        n_parts = len(parts)
        op_sort_key = None if self.acquaintance_size is None else lambda op: min((qubit_to_position[q] for q in op.qubits)) % self.acquaintance_size
        layers = new_layers()
        for layer_num in range(n_parts):
            layers = new_layers(prior_interstitial=layers.posterior_interstitial)
            for i in range(layer_num % 2, n_parts - 1, 2):
                left_part, right_part = parts[i:i + 2]
                acquaint_and_shift(parts=(left_part, right_part), layers=layers, acquaintance_size=self.acquaintance_size, swap_gate=self.swap_gate, mapping=mapping)
                parts_qubits = list(left_part + right_part)
                parts[i] = parts_qubits[:len(right_part)]
                parts[i + 1] = parts_qubits[len(right_part):]
            layers.prior_interstitial.sort(key=op_sort_key)
            for l in ('prior_interstitial', 'pre', 'intra', 'post'):
                yield getattr(layers, l)
        layers.posterior_interstitial.sort(key=op_sort_key)
        yield layers.posterior_interstitial
        assert list(itertools.chain(*(sorted((mapping[q] for q in part)) for part in reversed(parts)))) == list(range(n_qubits))
        final_permutation = {i: n_qubits - 1 - mapping[q] for i, q in enumerate(qubits)}
        final_gate = LinearPermutationGate(n_qubits, final_permutation, self.swap_gate)
        if final_gate:
            yield final_gate(*qubits)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        wire_symbol = '×' if args.use_unicode_characters else 'swap'
        wire_symbols = tuple((wire_symbol + f'({part_index},{qubit_index})' for part_index, part_len in enumerate(self.part_lens) for qubit_index in range(part_len)))
        return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols)

    @staticmethod
    def from_operations(qubit_order: Sequence['cirq.Qid'], operations: Sequence['cirq.Operation'], acquaintance_size: Optional[int]=0, swap_gate: 'cirq.Gate'=ops.SWAP) -> 'SwapNetworkGate':
        part_sizes = operations_to_part_lens(qubit_order, operations)
        return SwapNetworkGate(part_sizes, acquaintance_size, swap_gate)

    def permutation(self) -> Dict[int, int]:
        return {i: j for i, j in enumerate(reversed(range(sum(self.part_lens))))}

    def __repr__(self) -> str:
        return f'cirq.contrib.acquaintance.SwapNetworkGate({self.part_lens!r}, {self.acquaintance_size!r})'

    def _value_equality_values_(self):
        return (self.part_lens, self.acquaintance_size, self.swap_gate)
from functools import reduce
from typing import List, NamedTuple, Sequence, Tuple
from dataclasses import dataclass
import numpy as np
import cirq
from cirq import value
from cirq._compat import proper_repr, proper_eq
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
@dataclass
class TwoQubitGateTabulation:
    """A 2-qubit gate compiler based on precomputing/tabulating gate products."""
    base_gate: np.ndarray
    kak_vecs: np.ndarray
    single_qubit_gates: Sequence[Sequence[_SingleQubitGatePair]]
    max_expected_infidelity: float
    summary: str
    missed_points: Tuple[np.ndarray, ...]

    def compile_two_qubit_gate(self, unitary: np.ndarray) -> TwoQubitGateTabulationResult:
        """Compute single qubit gates required to compile a desired unitary.

        Given a desired unitary U, this computes the sequence of 1-local gates
        $k_j$ such that the product

        $k_{n-1} A k_{n-2} A ... k_1 A k_0$

        is close to U. Here A is the base_gate of the tabulation.

        Args:
            unitary: Unitary (U above) to compile.

        Returns:
            A TwoQubitGateTabulationResult object encoding the required local
            unitaries and resulting product above.
        """
        unitary = np.asarray(unitary)
        kak_vec = cirq.kak_vector(unitary, check_preconditions=False)
        infidelities = kak_vector_infidelity(kak_vec, self.kak_vecs, ignore_equivalent_vectors=True)
        nearest_ind = int(infidelities.argmin())
        success = infidelities[nearest_ind] < self.max_expected_infidelity
        inner_gates = np.array(self.single_qubit_gates[nearest_ind])
        if inner_gates.size == 0:
            kR, kL, actual = _outer_locals_for_unitary(unitary, self.base_gate)
            return TwoQubitGateTabulationResult(self.base_gate, unitary, (kR, kL), actual, success)
        inner_gates = vector_kron(inner_gates[..., 0, :, :], inner_gates[..., 1, :, :])
        assert inner_gates.ndim == 3
        inner_product = reduce(lambda a, b: self.base_gate @ b @ a, inner_gates, self.base_gate)
        kR, kL, actual = _outer_locals_for_unitary(unitary, inner_product)
        out = [kR]
        out.extend(self.single_qubit_gates[nearest_ind])
        out.append(kL)
        return TwoQubitGateTabulationResult(self.base_gate, unitary, tuple(out), actual, success)

    def _json_dict_(self):
        return {'base_gate': self.base_gate.tolist(), 'kak_vecs': self.kak_vecs.tolist(), 'single_qubit_gates': self.single_qubit_gates, 'max_expected_infidelity': self.max_expected_infidelity, 'summary': self.summary, 'missed_points': self.missed_points}

    def __repr__(self) -> str:
        numpy_single_qubit_gates = []
        for single_qubit_gate in self.single_qubit_gates:
            gate_repr = [f'({proper_repr(pair[0])}, {proper_repr(pair[1])})' for pair in single_qubit_gate]
            numpy_single_qubit_gates.append(f'[{','.join(gate_repr)}]')
        return f'cirq.TwoQubitGateTabulation({proper_repr(self.base_gate)}, {proper_repr(self.kak_vecs)}, [{','.join(numpy_single_qubit_gates)}],  {proper_repr(self.max_expected_infidelity)}, {proper_repr(self.summary)}, {proper_repr(self.missed_points)})'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.array_equal(self.base_gate, other.base_gate) and np.array_equal(self.kak_vecs, other.kak_vecs) and proper_eq(self.single_qubit_gates, other.single_qubit_gates) and (self.max_expected_infidelity == other.max_expected_infidelity) and (self.summary == other.summary) and np.array_equal(self.missed_points, other.missed_points)

    @classmethod
    def _from_json_dict_(cls, base_gate, kak_vecs, single_qubit_gates, max_expected_infidelity, summary, missed_points, **kwargs):
        numpy_single_qubit_gates = []
        for single_qubit_gate in single_qubit_gates:
            numpy_single_qubit_gate = []
            for pair in single_qubit_gate:
                numpy_tuple = (np.array(pair[0]), np.array(pair[1]))
                numpy_single_qubit_gate.append(numpy_tuple)
            numpy_single_qubit_gates.append(numpy_single_qubit_gate)
        return cls(base_gate=np.array(base_gate), kak_vecs=np.array(kak_vecs), single_qubit_gates=numpy_single_qubit_gates, max_expected_infidelity=max_expected_infidelity, summary=summary, missed_points=missed_points)
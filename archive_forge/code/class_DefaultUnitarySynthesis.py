from __future__ import annotations
from math import pi, inf, isclose
from typing import Any
from copy import deepcopy
from itertools import product
from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.synthesis.two_qubit.xx_decompose import XXDecomposer, XXEmbodiments
from qiskit.synthesis.two_qubit.two_qubit_decompose import (
from qiskit.quantum_info import Operator
from qiskit.circuit import ControlFlowOp, Gate, Parameter
from qiskit.circuit.library.standard_gates import (
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
from qiskit.providers.models import BackendProperties
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
class DefaultUnitarySynthesis(plugin.UnitarySynthesisPlugin):
    """The default unitary synthesis plugin."""

    @property
    def supports_basis_gates(self):
        return True

    @property
    def supports_coupling_map(self):
        return True

    @property
    def supports_natural_direction(self):
        return True

    @property
    def supports_pulse_optimize(self):
        return True

    @property
    def supports_gate_lengths(self):
        return False

    @property
    def supports_gate_errors(self):
        return False

    @property
    def supports_gate_lengths_by_qubit(self):
        return True

    @property
    def supports_gate_errors_by_qubit(self):
        return True

    @property
    def max_qubits(self):
        return None

    @property
    def min_qubits(self):
        return None

    @property
    def supported_bases(self):
        return None

    @property
    def supports_target(self):
        return True

    def __init__(self):
        super().__init__()
        self._decomposer_cache = {}

    def _decomposer_2q_from_target(self, target, qubits, approximation_degree):
        qubits_tuple = tuple(sorted(qubits))
        reverse_tuple = qubits_tuple[::-1]
        if qubits_tuple in self._decomposer_cache:
            return self._decomposer_cache[qubits_tuple]
        available_2q_basis = {}
        available_2q_props = {}

        def _replace_parameterized_gate(op):
            if isinstance(op, RXXGate) and isinstance(op.params[0], Parameter):
                op = RXXGate(pi / 2)
            elif isinstance(op, RZXGate) and isinstance(op.params[0], Parameter):
                op = RZXGate(pi / 4)
            return op
        try:
            keys = target.operation_names_for_qargs(qubits_tuple)
            for key in keys:
                op = target.operation_from_name(key)
                if not isinstance(op, Gate):
                    continue
                available_2q_basis[key] = _replace_parameterized_gate(op)
                available_2q_props[key] = target[key][qubits_tuple]
        except KeyError:
            pass
        try:
            keys = target.operation_names_for_qargs(reverse_tuple)
            for key in keys:
                if key not in available_2q_basis:
                    op = target.operation_from_name(key)
                    if not isinstance(op, Gate):
                        continue
                    available_2q_basis[key] = _replace_parameterized_gate(op)
                    available_2q_props[key] = target[key][reverse_tuple]
        except KeyError:
            pass
        if not available_2q_basis:
            raise TranspilerError(f'Target has no gates available on qubits {qubits} to synthesize over.')
        available_1q_basis = _find_matching_euler_bases(target, qubits_tuple[0])
        decomposers = []

        def is_supercontrolled(gate):
            try:
                operator = Operator(gate)
            except QiskitError:
                return False
            kak = TwoQubitWeylDecomposition(operator.data)
            return isclose(kak.a, pi / 4) and isclose(kak.c, 0.0)

        def is_controlled(gate):
            try:
                operator = Operator(gate)
            except QiskitError:
                return False
            kak = TwoQubitWeylDecomposition(operator.data)
            return isclose(kak.b, 0.0) and isclose(kak.c, 0.0)
        supercontrolled_basis = {k: v for k, v in available_2q_basis.items() if is_supercontrolled(v)}
        for basis_1q, basis_2q in product(available_1q_basis, supercontrolled_basis.keys()):
            props = available_2q_props.get(basis_2q)
            if props is None:
                basis_2q_fidelity = 1.0
            else:
                error = getattr(props, 'error', 0.0)
                if error is None:
                    error = 0.0
                basis_2q_fidelity = 1 - error
            if approximation_degree is not None:
                basis_2q_fidelity *= approximation_degree
            decomposer = TwoQubitBasisDecomposer(supercontrolled_basis[basis_2q], euler_basis=basis_1q, basis_fidelity=basis_2q_fidelity)
            decomposers.append(decomposer)
        controlled_basis = {k: v for k, v in available_2q_basis.items() if is_controlled(v)}
        basis_2q_fidelity = {}
        embodiments = {}
        pi2_basis = None
        for k, v in controlled_basis.items():
            strength = 2 * TwoQubitWeylDecomposition(Operator(v).data).a
            props = available_2q_props.get(k)
            if props is None:
                basis_2q_fidelity[strength] = 1.0
            else:
                error = getattr(props, 'error', 0.0)
                if error is None:
                    error = 0.0
                basis_2q_fidelity[strength] = 1 - error
            embodiment = XXEmbodiments[v.base_class]
            if len(embodiment.parameters) == 1:
                embodiments[strength] = embodiment.assign_parameters([strength])
            else:
                embodiments[strength] = embodiment
            if isclose(strength, pi / 2) and k in supercontrolled_basis:
                pi2_basis = v
        if approximation_degree is not None:
            basis_2q_fidelity = {k: v * approximation_degree for k, v in basis_2q_fidelity.items()}
        if basis_2q_fidelity:
            for basis_1q in available_1q_basis:
                if isinstance(pi2_basis, CXGate) and basis_1q == 'ZSX':
                    pi2_decomposer = TwoQubitBasisDecomposer(pi2_basis, euler_basis=basis_1q, basis_fidelity=basis_2q_fidelity, pulse_optimize=True)
                    embodiments.update({pi / 2: XXEmbodiments[pi2_decomposer.gate.base_class]})
                else:
                    pi2_decomposer = None
                decomposer = XXDecomposer(basis_fidelity=basis_2q_fidelity, euler_basis=basis_1q, embodiments=embodiments, backup_optimizer=pi2_decomposer)
                decomposers.append(decomposer)
        self._decomposer_cache[qubits_tuple] = decomposers
        return decomposers

    def run(self, unitary, **options):
        approximation_degree = getattr(self, '_approximation_degree', 1.0)
        basis_gates = options['basis_gates']
        coupling_map = options['coupling_map'][0]
        natural_direction = options['natural_direction']
        pulse_optimize = options['pulse_optimize']
        gate_lengths = options['gate_lengths_by_qubit']
        gate_errors = options['gate_errors_by_qubit']
        qubits = options['coupling_map'][1]
        target = options['target']
        if unitary.shape == (2, 2):
            _decomposer1q = Optimize1qGatesDecomposition(basis_gates, target)
            sequence = _decomposer1q._resynthesize_run(unitary, qubits[0])
            if sequence is None:
                return None
            return _decomposer1q._gate_sequence_to_dag(sequence)
        elif unitary.shape == (4, 4):
            if target is not None:
                decomposers2q = self._decomposer_2q_from_target(target, qubits, approximation_degree)
            else:
                decomposer2q = _decomposer_2q_from_basis_gates(basis_gates, pulse_optimize, approximation_degree)
                decomposers2q = [decomposer2q] if decomposer2q is not None else []
            synth_circuits = []
            for decomposer2q in decomposers2q:
                preferred_direction = _preferred_direction(decomposer2q, qubits, natural_direction, coupling_map, gate_lengths, gate_errors)
                synth_circuit = self._synth_su4(unitary, decomposer2q, preferred_direction, approximation_degree)
                synth_circuits.append(synth_circuit)
            synth_circuit = min(synth_circuits, key=partial(_error, target=target, qubits=tuple(qubits)), default=None)
        else:
            from qiskit.synthesis.unitary.qsd import qs_decomposition
            synth_circuit = qs_decomposition(unitary) if basis_gates or target else None
        synth_dag = circuit_to_dag(synth_circuit) if synth_circuit is not None else None
        return synth_dag

    def _synth_su4(self, su4_mat, decomposer2q, preferred_direction, approximation_degree):
        approximate = not approximation_degree == 1.0
        synth_circ = decomposer2q(su4_mat, approximate=approximate)
        synth_direction = None
        for inst in synth_circ:
            if inst.operation.num_qubits == 2:
                synth_direction = [synth_circ.find_bit(q).index for q in inst.qubits]
        if preferred_direction and synth_direction != preferred_direction:
            su4_mat_mm = deepcopy(su4_mat)
            su4_mat_mm[[1, 2]] = su4_mat_mm[[2, 1]]
            su4_mat_mm[:, [1, 2]] = su4_mat_mm[:, [2, 1]]
            synth_circ = decomposer2q(su4_mat_mm, approximate=approximate).reverse_bits()
        return synth_circ
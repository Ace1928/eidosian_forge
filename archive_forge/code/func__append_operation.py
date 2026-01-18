from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_operation(clifford, operation, qargs=None):
    """Update Clifford inplace by applying a Clifford operation.

    Args:
        clifford (Clifford): The Clifford to update.
        operation (Instruction or Clifford or str): The operation or composite operation to apply.
        qargs (list or None): The qubits to apply operation to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input operation cannot be converted into Clifford operations.
    """
    if isinstance(operation, (Barrier, Delay)):
        return clifford
    if qargs is None:
        qargs = list(range(clifford.num_qubits))
    gate = operation
    if isinstance(gate, str):
        if gate not in _BASIS_1Q and gate not in _BASIS_2Q:
            raise QiskitError(f'Invalid Clifford gate name string {gate}')
        name = gate
    else:
        name = gate.name
        if getattr(gate, 'condition', None) is not None:
            raise QiskitError('Conditional gate is not a valid Clifford operation.')
    if name in _NON_CLIFFORD:
        raise QiskitError(f'Cannot update Clifford with non-Clifford gate {name}')
    if name in _BASIS_1Q:
        if len(qargs) != 1:
            raise QiskitError('Invalid qubits for 1-qubit gate.')
        return _BASIS_1Q[name](clifford, qargs[0])
    if name in _BASIS_2Q:
        if len(qargs) != 2:
            raise QiskitError('Invalid qubits for 2-qubit gate.')
        return _BASIS_2Q[name](clifford, qargs[0], qargs[1])
    if isinstance(gate, Gate) and name == 'u' and (len(qargs) == 1):
        try:
            theta, phi, lambd = tuple((_n_half_pis(par) for par in gate.params))
        except ValueError as err:
            raise QiskitError('U gate angles must be multiples of pi/2 to be a Clifford') from err
        if theta == 0:
            clifford = _append_rz(clifford, qargs[0], lambd + phi)
        elif theta == 1:
            clifford = _append_rz(clifford, qargs[0], lambd - 2)
            clifford = _append_h(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi)
        elif theta == 2:
            clifford = _append_rz(clifford, qargs[0], lambd - 1)
            clifford = _append_x(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi + 1)
        elif theta == 3:
            clifford = _append_rz(clifford, qargs[0], lambd)
            clifford = _append_h(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi + 2)
        return clifford
    from qiskit.quantum_info import Clifford
    if isinstance(gate, Clifford):
        composed_clifford = clifford.compose(gate, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford
    from qiskit.circuit.library import LinearFunction
    if isinstance(gate, LinearFunction):
        gate_as_clifford = Clifford.from_linear_function(gate)
        composed_clifford = clifford.compose(gate_as_clifford, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford
    from qiskit.circuit.library import PermutationGate
    if isinstance(gate, PermutationGate):
        gate_as_clifford = Clifford.from_permutation(gate)
        composed_clifford = clifford.compose(gate_as_clifford, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford
    if gate.definition is not None:
        try:
            return _append_circuit(clifford.copy(), gate.definition, qargs)
        except QiskitError:
            pass
    if isinstance(gate, Gate) and len(qargs) <= 3:
        try:
            matrix = gate.to_matrix()
            gate_cliff = Clifford.from_matrix(matrix)
            return _append_operation(clifford, gate_cliff, qargs=qargs)
        except TypeError as err:
            raise QiskitError(f'Cannot apply {gate.name} gate with unbounded parameters') from err
        except CircuitError as err:
            raise QiskitError(f'Cannot apply {gate.name} gate without to_matrix defined') from err
        except QiskitError as err:
            raise QiskitError(f'Cannot apply non-Clifford gate: {gate.name}') from err
    raise QiskitError(f'Cannot apply {gate}')
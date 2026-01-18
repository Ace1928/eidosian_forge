import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _pennylane_to_openfermion(coeffs, ops, wires=None):
    """Convert a 2-tuple of complex coefficients and PennyLane operations to
    OpenFermion ``QubitOperator``.

    Args:
        coeffs (array[complex]):
            coefficients for each observable, same length as ops
        ops (Iterable[pennylane.operation.Operations]): list of PennyLane operations that
            have a valid PauliSentence representation.
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert to qubit operator
            from an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only consecutive-int-valued dict (for wire-to-qubit conversion) is
            accepted. If None, will map sorted wires from all `ops` to consecutive int.

    Returns:
        QubitOperator: an instance of OpenFermion's ``QubitOperator``.

    **Example**

    >>> coeffs = np.array([0.1, 0.2, 0.3, 0.4])
    >>> ops = [
    ...     qml.operation.Tensor(qml.X('w0')),
    ...     qml.operation.Tensor(qml.Y('w0'), qml.Z('w2')),
    ...     qml.sum(qml.Z('w0'), qml.s_prod(-0.5, qml.X('w0'))),
    ...     qml.prod(qml.X('w0'), qml.Z('w1')),
    ... ]
    >>> _pennylane_to_openfermion(coeffs, ops, wires=Wires(['w0', 'w1', 'w2']))
    (-0.05+0j) [X0] +
    (0.4+0j) [X0 Z1] +
    (0.2+0j) [Y0 Z2] +
    (0.3+0j) [Z0]
    """
    try:
        import openfermion
    except ImportError as Error:
        raise ImportError('This feature requires openfermion. It can be installed with: pip install openfermion') from Error
    all_wires = Wires.all_wires([op.wires for op in ops], sort=True)
    if wires is not None:
        qubit_indexed_wires = _process_wires(wires)
        if not set(all_wires).issubset(set(qubit_indexed_wires)):
            raise ValueError('Supplied `wires` does not cover all wires defined in `ops`.')
    else:
        qubit_indexed_wires = all_wires
    q_op = openfermion.QubitOperator()
    for coeff, op in zip(coeffs, ops):
        if isinstance(op, Tensor):
            try:
                ps = pauli_sentence(op)
            except ValueError as e:
                raise ValueError(f'Expected a Pennylane operator with a valid Pauli word representation, but got {op}.') from e
        elif (ps := op.pauli_rep) is None:
            raise ValueError(f'Expected a Pennylane operator with a valid Pauli word representation, but got {op}.')
        if len(ps) > 0:
            sub_coeffs, op_strs = _ps_to_coeff_term(ps, wire_order=qubit_indexed_wires)
            for c, op_str in zip(sub_coeffs, op_strs):
                q_op += complex(coeff * c) * openfermion.QubitOperator(op_str)
    return q_op
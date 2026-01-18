import warnings
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires
def _openfermion_to_pennylane(qubit_operator, wires=None):
    """Convert OpenFermion ``QubitOperator`` to a 2-tuple of coefficients and
    PennyLane Pauli observables.

    Args:
        qubit_operator (QubitOperator): fermionic-to-qubit transformed operator in terms of
            Pauli matrices
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        tuple[array[float], Iterable[pennylane.operation.Operator]]: coefficients and their
        corresponding PennyLane observables in the Pauli basis

    **Example**

    >>> q_op = 0.1*QubitOperator('X0') + 0.2*QubitOperator('Y0 Z2')
    >>> q_op
    0.1 [X0] +
    0.2 [Y0 Z2]
    >>> _openfermion_to_pennylane(q_op, wires=['w0','w1','w2','extra_wire'])
    (tensor([0.1, 0.2], requires_grad=False), [X('w0'), Y('w0') @ Z('w2')])

    If the new op-math is active, the list of operators will be cast as :class:`~.Prod` instances instead of
    :class:`~.Tensor` instances when appropriate.
    """
    n_wires = 1 + max((max((i for i, _ in t)) if t else 1 for t in qubit_operator.terms)) if qubit_operator.terms else 1
    wires = _process_wires(wires, n_wires=n_wires)
    if not qubit_operator.terms:
        return (np.array([0.0]), [qml.Identity(wires[0])])
    xyz2pauli = {'X': qml.X, 'Y': qml.Y, 'Z': qml.Z}

    def _get_op(term, wires):
        """A function to compute the PL operator associated with the term string."""
        if len(term) > 1:
            if active_new_opmath():
                return qml.prod(*[xyz2pauli[op[1]](wires=wires[op[0]]) for op in term])
            return Tensor(*[xyz2pauli[op[1]](wires=wires[op[0]]) for op in term])
        if len(term) == 1:
            return xyz2pauli[term[0][1]](wires=wires[term[0][0]])
        return qml.Identity(wires[0])
    coeffs, ops = zip(*[(coef, _get_op(term, wires)) for term, coef in qubit_operator.terms.items()])
    return (np.array(coeffs).real, list(ops))
from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def _check_eigendecomposition(op):
    """Checks involving diagonalizing gates and eigenvalues."""
    if op.has_diagonalizing_gates:
        dg = op.diagonalizing_gates()
        try:
            compute_dg = type(op).compute_diagonalizing_gates(*op.data, wires=op.wires, **op.hyperparameters)
        except (qml.operation.DiagGatesUndefinedError, TypeError):
            compute_dg = dg
        for op1, op2 in zip(dg, compute_dg):
            assert op1 == op2, 'diagonalizing_gates and compute_diagonalizing_gates must match'
    else:
        failure_comment = 'If has_diagonalizing_gates is False, diagonalizing_gates must raise a DiagGatesUndefinedError'
        _assert_error_raised(op.diagonalizing_gates, qml.operation.DiagGatesUndefinedError, failure_comment)()
    try:
        eg = op.eigvals()
    except EigvalsUndefinedError:
        eg = None
    has_eigvals = True
    try:
        compute_eg = type(op).compute_eigvals(*op.data, **op.hyperparameters)
    except EigvalsUndefinedError:
        compute_eg = eg
        has_eigvals = False
    if has_eigvals:
        assert qml.math.allclose(eg, compute_eg), 'eigvals and compute_eigvals must match'
    if has_eigvals and op.has_diagonalizing_gates:
        dg = qml.prod(*dg[::-1]) if len(dg) > 0 else qml.Identity(op.wires)
        eg = qml.QubitUnitary(np.diag(eg), wires=op.wires)
        decomp = qml.prod(qml.adjoint(dg), eg, dg)
        decomp_mat = qml.matrix(decomp)
        original_mat = qml.matrix(op)
        failure_comment = f'eigenvalues and diagonalizing gates must be able to reproduce the original operator. Got \n{decomp_mat}\n\n{original_mat}'
        assert qml.math.allclose(decomp_mat, original_mat), failure_comment
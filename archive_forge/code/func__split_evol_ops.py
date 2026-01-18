from typing import Sequence, Callable
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution, HardwareHamiltonian
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .general_shift_rules import eigvals_to_frequencies, generate_shift_rule
from .gradient_transform import (
def _split_evol_ops(op, ob, tau):
    """Randomly split a ``ParametrizedEvolution`` with respect to time into two operations and
    insert a Pauli rotation using a given Pauli word and rotation angles :math:`\\pm\\pi/2`.
    This yields two groups of three operations each.

    Args:
        op (ParametrizedEvolution): operation to split up.
        ob (`~.Operator`): generating Hamiltonian term to insert the parameter-shift rule for.
        tau (float or tensor_like): split-up time(s). If multiple times are passed, the split-up
            operations are set up to return intermediate time evolution results, leading to
            broadcasting effectively.

    Returns:
        tuple[list[`~.Operation`]]: The split-time evolution, expressed as three operations in the
            inner lists. The number of tuples is given by the number of shifted terms in the
            parameter-shift rule of the generating Hamiltonian term ``ob``.
        tensor_like: Coefficients of the parameter-shift rule of the provided generating Hamiltonian
            term ``ob``.
    """
    t0, *_, t1 = op.t
    if (bcast := (qml.math.ndim(tau) > 0)):
        tau = jnp.sort(tau)
        before_t = jnp.concatenate([jnp.array([t0]), tau, jnp.array([t1])])
        after_t = before_t.copy()
    else:
        before_t = jax.numpy.array([t0, tau])
        after_t = jax.numpy.array([tau, t1])
    if qml.pauli.is_pauli_word(ob):
        prefactor = qml.pauli.pauli_word_prefactor(ob)
        word = qml.pauli.pauli_word_to_string(ob)
        insert_ops = [qml.PauliRot(shift, word, ob.wires) for shift in [np.pi / 2, -np.pi / 2]]
        coeffs = [prefactor, -prefactor]
    else:
        with warnings.catch_warnings():
            if len(ob.wires) <= 4:
                warnings.filterwarnings('ignore', '.*the eigenvalues will be computed numerically.*')
            eigvals = qml.eigvals(ob)
        coeffs, shifts = zip(*generate_shift_rule(eigvals_to_frequencies(tuple(eigvals))))
        insert_ops = [qml.exp(qml.dot([-1j * shift], [ob])) for shift in shifts]
    ode_kwargs = op.odeint_kwargs
    ops = tuple(([op(op.data, before_t, return_intermediate=bcast, **ode_kwargs), insert_op, op(op.data, after_t, return_intermediate=bcast, complementary=bcast, **ode_kwargs)] for insert_op in insert_ops))
    return (ops, jnp.array(coeffs))
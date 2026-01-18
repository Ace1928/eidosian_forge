from functools import singledispatch
from string import ascii_letters as alphabet
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.measurements import MidMeasureMP
from pennylane.ops import Conditional
def _evolve_state_vector_under_parametrized_evolution(operation: qml.pulse.ParametrizedEvolution, state, num_wires, is_state_batched):
    """Uses an odeint solver to compute the evolution of the input ``state`` under the given
    ``ParametrizedEvolution`` operation.

    Args:
        state (array[complex]): input state
        operation (ParametrizedEvolution): operation to apply on the state

    Raises:
        ValueError: If the parameters and time windows of the ``ParametrizedEvolution`` are
            not defined.

    Returns:
        TensorLike[complex]: output state
    """
    try:
        import jax
        from jax.experimental.ode import odeint
        from pennylane.pulse.parametrized_hamiltonian_pytree import ParametrizedHamiltonianPytree
    except ImportError as e:
        raise ImportError('Module jax is required for the ``ParametrizedEvolution`` class. You can install jax via: pip install jax') from e
    if operation.data is None or operation.t is None:
        raise ValueError('The parameters and the time window are required to execute a ParametrizedEvolution You can update these values by calling the ParametrizedEvolution class: EV(params, t).')
    if is_state_batched:
        batch_dim = state.shape[0]
        state = qml.math.moveaxis(state.reshape((batch_dim, 2 ** num_wires)), 1, 0)
        out_shape = [2] * num_wires + [batch_dim]
    else:
        state = state.flatten()
        out_shape = [2] * num_wires
    with jax.ensure_compile_time_eval():
        H_jax = ParametrizedHamiltonianPytree.from_hamiltonian(operation.H, dense=operation.dense, wire_order=list(np.arange(num_wires)))

    def fun(y, t):
        """dy/dt = -i H(t) y"""
        return -1j * H_jax(operation.data, t=t) @ y
    result = odeint(fun, state, operation.t, **operation.odeint_kwargs)
    if operation.hyperparameters['return_intermediate']:
        return qml.math.reshape(result, [-1] + out_shape)
    result = qml.math.reshape(result[-1], out_shape)
    if is_state_batched:
        return qml.math.moveaxis(result, -1, 0)
    return result
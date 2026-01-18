from numbers import Number
from typing import Tuple
import numpy as np
import pennylane as qml
from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape
from .apply_operation import apply_operation
from .simulate import get_final_state
from .initialize_state import create_initial_state
def _get_vjp_bras(tape, cotangents, ket):
    """Helper function for getting the bras for adjoint vjp, the batch size of the
    cotangents, as well as a list of indices for which the cotangents are zero.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of.
        tangents (Tuple[Number]): gradient vector for input parameters.
        ket (TensorLike): the final state of the circuit.

    Returns:
        Tuple[TensorLike, int, List]: The return contains the following:
            * Final bra for batch size ``None``, else array of bras
            * Batch size. None if cotangents are not batched
            * List containing batch indices that are zero. Empty for unbatched
              cotangents
    """
    if isinstance(tape.measurements[0], qml.measurements.StateMP):
        batched_cotangents = np.ndim(cotangents) == 2
        batch_size = np.shape(cotangents)[0] if batched_cotangents else None
        bras = np.conj(cotangents.reshape(-1, *ket.shape))
        bras = bras if batched_cotangents else np.squeeze(bras)
        return (bras, batch_size, [])
    single_cotangent = len(tape.measurements) == 1
    if not single_cotangent:
        inner_shape = next((np.shape(cot) for cot in cotangents if np.shape(cot) != ()), None)
        if inner_shape is not None:
            new_cotangents = []
            for i, c in enumerate(cotangents):
                if np.shape(c) == () and np.allclose(c, 0.0):
                    new_cotangents.append(np.zeros(inner_shape))
                else:
                    new_cotangents.append(c)
            cotangents = new_cotangents
    cotangents = np.array(cotangents)
    if single_cotangent:
        cotangents = np.expand_dims(cotangents, 0)
    batched_cotangents = np.ndim(cotangents) == 2
    batch_size = cotangents.shape[1] if batched_cotangents else None
    if np.allclose(cotangents, 0.0):
        return (None, batch_size, [])
    new_obs, null_batch_indices = ([], [])
    if batched_cotangents:
        for i, cots in enumerate(cotangents.T):
            new_cs, new_os = ([], [])
            for c, o in zip(cots, tape.observables):
                if not np.allclose(c, 0.0):
                    new_cs.append(c)
                    new_os.append(o)
            if len(new_cs) == 0:
                null_batch_indices.append(i)
            else:
                new_obs.append(qml.dot(new_cs, new_os))
    else:
        new_cs, new_os = ([], [])
        for c, o in zip(cotangents, tape.observables):
            if not np.allclose(c, 0.0):
                new_cs.append(c)
                new_os.append(o)
        new_obs.append(qml.dot(new_cs, new_os))
    bras = np.empty((len(new_obs), *ket.shape), dtype=ket.dtype)
    for kk, obs in enumerate(new_obs):
        if obs.pauli_rep is not None:
            flat_bra = obs.pauli_rep.dot(ket.flatten(), wire_order=list(range(tape.num_wires)))
            bras[kk] = 2 * flat_bra.reshape(ket.shape)
        else:
            bras[kk] = 2 * apply_operation(obs, ket)
    bras = bras if batched_cotangents else np.squeeze(bras)
    return (bras, batch_size, null_batch_indices)
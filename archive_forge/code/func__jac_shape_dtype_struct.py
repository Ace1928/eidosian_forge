from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def _jac_shape_dtype_struct(tape: 'qml.tape.QuantumScript', device: 'qml.Device'):
    """The shape of a jacobian for a single tape given a device.

    Args:
        tape (QuantumTape): the tape who's output we want to determine
        device (Device): the device used to execute the tape.

    >>> tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.X(0)), qml.probs(0)])
    >>> dev = qml.devices.DefaultQubit()
    >>> _jac_shape_dtype_struct(tape, dev)
    (ShapeDtypeStruct(shape=(), dtype=float64),
    ShapeDtypeStruct(shape=(2,), dtype=float64))
    >>> tapes, fn = qml.gradients.param_shift(tape)
    >>> fn(dev.execute(tapes))
    (array(0.), array([-0.42073549,  0.42073549]))
    """
    shape_and_dtype = _result_shape_dtype_struct(tape, device)
    if len(tape.trainable_params) == 1:
        return shape_and_dtype
    if len(tape.measurements) == 1:
        return tuple((shape_and_dtype for _ in tape.trainable_params))
    return tuple((tuple((_s for _ in tape.trainable_params)) for _s in shape_and_dtype))
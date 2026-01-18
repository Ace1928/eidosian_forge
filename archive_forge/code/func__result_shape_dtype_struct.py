from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def _result_shape_dtype_struct(tape: 'qml.tape.QuantumScript', device: 'qml.Device'):
    """Auxiliary function for creating the shape and dtype object structure
    given a tape."""
    shape = tape.shape(device)
    if len(tape.measurements) == 1:
        m_dtype = _jax_dtype(tape.measurements[0].numeric_type)
        if tape.shots.has_partitioned_shots:
            return tuple((jax.ShapeDtypeStruct(s, m_dtype) for s in shape))
        return jax.ShapeDtypeStruct(tuple(shape), m_dtype)
    tape_dtype = tuple((_jax_dtype(m.numeric_type) for m in tape.measurements))
    if tape.shots.has_partitioned_shots:
        return tuple((tuple((jax.ShapeDtypeStruct(tuple(s), d) for s, d in zip(si, tape_dtype))) for si in shape))
    return tuple((jax.ShapeDtypeStruct(tuple(s), d) for s, d in zip(shape, tape_dtype)))
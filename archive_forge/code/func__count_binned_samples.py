import numpy as np
import pennylane as qml
from pennylane.devices import DefaultQubitLegacy
from pennylane.pulse import ParametrizedEvolution
from pennylane.typing import TensorLike
@staticmethod
def _count_binned_samples(indices, batch_size, dim, bin_size, num_bins):
    """Count the occurences of bins of sampled indices and convert them to relative
        counts in order to estimate their occurence probability per bin."""
    shape = (dim + 1, num_bins) if batch_size is None else (batch_size, dim + 1, num_bins)
    prob = qml.math.convert_like(jnp.zeros(shape, dtype=jnp.float64), indices)
    if batch_size is None:
        indices = indices.reshape((num_bins, bin_size))
        for b, idx in enumerate(indices):
            idx = qml.math.convert_like(idx, indices)
            basis_states, counts = jnp.unique(idx, return_counts=True, size=dim, fill_value=-1)
            for state, count in zip(basis_states, counts):
                prob = prob.at[state, b].set(count / bin_size)
        return prob[:-1]
    indices = indices.reshape((batch_size, num_bins, bin_size))
    for i, _indices in enumerate(indices):
        for b, idx in enumerate(_indices):
            idx = qml.math.convert_like(idx, indices)
            basis_states, counts = jnp.unique(idx, return_counts=True, size=dim, fill_value=-1)
            for state, count in zip(basis_states, counts):
                prob = prob.at[i, state, b].set(count / bin_size)
    return prob[:, :-1]
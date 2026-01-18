from typing import Sequence, Tuple
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import Probability, SampleMeasurement, StateMeasurement
from .mid_measure import MeasurementValue
@staticmethod
def _count_samples(indices, batch_size, dim):
    """Count the occurrences of sampled indices and convert them to relative
        counts in order to estimate their occurrence probability."""
    num_bins, bin_size = indices.shape[-2:]
    if batch_size is None:
        prob = qml.math.zeros((dim, num_bins), dtype='float64')
        for b, idx in enumerate(indices):
            basis_states, counts = qml.math.unique(idx, return_counts=True)
            prob[basis_states, b] = counts / bin_size
        return prob
    prob = qml.math.zeros((batch_size, dim, num_bins), dtype='float64')
    indices = indices.reshape((batch_size, num_bins, bin_size))
    for i, _indices in enumerate(indices):
        for b, idx in enumerate(_indices):
            basis_states, counts = qml.math.unique(idx, return_counts=True)
            prob[i, basis_states, b] = counts / bin_size
    return prob
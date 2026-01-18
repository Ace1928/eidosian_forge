from __future__ import annotations
import copy
from abc import abstractmethod
import numpy as np
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts
@staticmethod
def _index_to_ket_array(inds: np.ndarray, dims: tuple, string_labels: bool=False) -> np.ndarray:
    """Convert an index array into a ket array.

        Args:
            inds (np.array): an integer index array.
            dims (tuple): a list of subsystem dimensions.
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            np.array: an array of ket strings if string_label=True, otherwise
                      an array of ket lists.
        """
    shifts = [1]
    for dim in dims[:-1]:
        shifts.append(shifts[-1] * dim)
    kets = np.array([inds // shift % dim for dim, shift in zip(dims, shifts)])
    if string_labels:
        max_dim = max(dims)
        char_kets = np.asarray(kets, dtype=np.str_)
        str_kets = char_kets[0]
        for row in char_kets[1:]:
            if max_dim > 10:
                str_kets = np.char.add(',', str_kets)
            str_kets = np.char.add(row, str_kets)
        return str_kets.T
    return kets.T
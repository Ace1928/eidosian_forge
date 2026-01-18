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
def _vector_to_dict(vec, dims, decimals=None, string_labels=False):
    """Convert a vector to a ket dictionary.

        This representation will not show zero values in the output dict.

        Args:
            vec (array): a Numpy vector array.
            dims (tuple): subsystem dimensions.
            decimals (None or int): number of decimal places to round to.
                                    (See Numpy.round), if None no rounding
                                    is done (Default: None).
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            dict: the vector in dictionary `ket` form.
        """
    vals = vec if decimals is None else vec.round(decimals=decimals)
    inds, = vals.nonzero()
    kets = QuantumState._index_to_ket_array(inds, dims, string_labels=string_labels)
    if string_labels:
        return dict(zip(kets, vec[inds]))
    return {tuple(ket): val for ket, val in zip(kets, vals[inds])}
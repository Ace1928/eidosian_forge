import string
from typing import List, Sequence
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import numpy as pnp
from .utils import MeasureNode, PrepareNode
def _process_tensor(results, n_prep: int, n_meas: int):
    """Convert a flat slice of an individual circuit fragment's execution results into a tensor.

    This function performs the following steps:

    1. Reshapes ``results`` into the intermediate shape ``(4,) * n_prep + (4**n_meas,)``
    2. Shuffles the final axis to follow the standard product over measurement settings. E.g., for
      ``n_meas = 2`` the standard product is: II, IX, IY, IZ, XI, ..., ZY, ZZ while the input order
      will be the result of ``qml.pauli.partition_pauli_group(2)``, i.e., II, IZ, ZI, ZZ, ...,
      YY.
    3. Reshapes into the final target shape ``(4,) * (n_prep + n_meas)``
    4. Performs a change of basis for the preparation indices (the first ``n_prep`` indices) from
       the |0>, |1>, |+>, |+i> basis to the I, X, Y, Z basis using ``CHANGE_OF_BASIS``.

    Args:
        results (tensor_like): the input execution results
        n_prep (int): the number of preparation nodes in the corresponding circuit fragment
        n_meas (int): the number of measurement nodes in the corresponding circuit fragment

    Returns:
        tensor_like: the corresponding fragment tensor
    """
    n = n_prep + n_meas
    dim_meas = 4 ** n_meas
    intermediate_shape = (4,) * n_prep + (dim_meas,)
    intermediate_tensor = qml.math.reshape(results, intermediate_shape)
    grouped = qml.pauli.partition_pauli_group(n_meas)
    grouped_flat = [term for group in grouped for term in group]
    order = qml.math.argsort(grouped_flat)
    if qml.math.get_interface(intermediate_tensor) == 'tensorflow':
        intermediate_tensor = qml.math.gather(intermediate_tensor, order, axis=-1)
    else:
        sl = [slice(None)] * n_prep + [order]
        intermediate_tensor = intermediate_tensor[tuple(sl)]
    final_shape = (4,) * n
    final_tensor = qml.math.reshape(intermediate_tensor, final_shape)
    change_of_basis = qml.math.convert_like(CHANGE_OF_BASIS, intermediate_tensor)
    for i in range(n_prep):
        axes = [[1], [i]]
        final_tensor = qml.math.tensordot(change_of_basis, final_tensor, axes=axes)
    axes = list(reversed(range(n_prep))) + list(range(n_prep, n))
    final_tensor = qml.math.transpose(final_tensor, axes=axes)
    final_tensor *= qml.math.power(2, -(n_meas + n_prep) / 2)
    return final_tensor
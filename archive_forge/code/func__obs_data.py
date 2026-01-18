import itertools
import numbers
from collections.abc import Iterable
from copy import copy
import functools
from typing import List
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import Observable, Tensor
from pennylane.wires import Wires
def _obs_data(self):
    """Extracts the data from a Hamiltonian and serializes it in an order-independent fashion.

        This allows for comparison between Hamiltonians that are equivalent, but are defined with terms and tensors
        expressed in different orders. For example, `qml.X(0) @ qml.Z(1)` and
        `qml.Z(1) @ qml.X(0)` are equivalent observables with different orderings.

        .. Note::

            In order to store the data from each term of the Hamiltonian in an order-independent serialization,
            we make use of sets. Note that all data contained within each term must be immutable, hence the use of
            strings and frozensets.

        **Example**

        >>> H = qml.Hamiltonian([1, 1], [qml.X(0) @ qml.X(1), qml.Z(0)])
        >>> print(H._obs_data())
        {(1, frozenset({('PauliX', <Wires = [1]>, ()), ('PauliX', <Wires = [0]>, ())})),
         (1, frozenset({('PauliZ', <Wires = [0]>, ())}))}
        """
    data = set()
    coeffs_arr = qml.math.toarray(self.coeffs)
    for co, op in zip(coeffs_arr, self.ops):
        obs = op.non_identity_obs if isinstance(op, Tensor) else [op]
        tensor = []
        for ob in obs:
            parameters = tuple((str(param) for param in ob.parameters))
            if isinstance(ob, qml.GellMann):
                parameters += (ob.hyperparameters['index'],)
            tensor.append((ob.name, ob.wires, parameters))
        data.add((co, frozenset(tensor)))
    return data
import itertools
import warnings
from copy import copy
from functools import reduce, wraps
from itertools import combinations
from typing import List, Tuple, Union
from scipy.sparse import kron as sparse_kron
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.op_math.pow import Pow
from pennylane.ops.op_math.sprod import SProd
from pennylane.ops.op_math.sum import Sum
from pennylane.ops.qubit import Hamiltonian
from pennylane.ops.qubit.non_parametric_ops import PauliX, PauliY, PauliZ
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .composite import CompositeOp
def _add_non_pauli_factor(self, factor: Operator, wires: List[int]):
    """Adds the given non-Pauli factor to the temporary ``self._non_pauli_factors`` dictionary.
        If there alerady exists an identical operator in the dictionary, the two are grouped
        together.

        If there isn't an identical operator in the dictionary, all non Pauli factors that act on
        the same wires are removed and added to the ``self._factors`` tuple.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
    if isinstance(factor, Pow):
        exponent = factor.z
        factor = factor.base
    else:
        exponent = 1
    op_hash = factor.hash
    old_hash, old_exponent, old_op = self._non_pauli_factors.get(wires, [None, None, None])
    if isinstance(old_op, (qml.RX, qml.RY, qml.RZ)) and factor.name == old_op.name:
        self._non_pauli_factors[wires] = [op_hash, old_exponent, factor.__class__(factor.data[0] + old_op.data[0], wires).simplify()]
    elif op_hash == old_hash:
        self._non_pauli_factors[wires][1] += exponent
    else:
        self._remove_non_pauli_factors(wires=wires)
        self._non_pauli_factors[wires] = [op_hash, copy(exponent), factor]
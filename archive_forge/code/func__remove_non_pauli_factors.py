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
def _remove_non_pauli_factors(self, wires: List[int]):
    """Remove all factors from the ``self._non_pauli_factors`` dictionary that act on the given
        wires and add them to the ``self._factors`` tuple.

        Args:
            wires (List[int]): Wires of the operators to be removed.
        """
    if not self._non_pauli_factors:
        return
    for wire in wires:
        for key, (_, exponent, op) in list(self._non_pauli_factors.items()):
            if wire in key:
                self._non_pauli_factors.pop(key)
                if exponent == 0:
                    continue
                if exponent != 1:
                    op = Pow(base=op, z=exponent).simplify()
                if not isinstance(op, qml.Identity):
                    self._factors += ((op,),)
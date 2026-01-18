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
def _add_pauli_factor(self, factor: Operator, wires: List[int]):
    """Adds the given Pauli operator to the temporary ``self._pauli_factors`` dictionary. If
        there was another Pauli operator acting on the same wire, the two operators are grouped
        together using the ``self._pauli_mult`` dictionary.

        Args:
            factor (Operator): Factor to be added.
            wires (List[int]): Factor wires. This argument is added to avoid calling
                ``factor.wires`` several times.
        """
    wire = wires[0]
    op2_name = factor.name
    old_coeff, old_word = self._pauli_factors.get(wire, (1, 'Identity'))
    coeff, new_word = self._pauli_mult[old_word][op2_name]
    self._pauli_factors[wire] = (old_coeff * coeff, new_word)
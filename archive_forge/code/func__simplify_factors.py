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
def _simplify_factors(self, factors: Tuple[Operator]) -> Tuple[complex, Operator]:
    """Reduces the depth of nested factors and groups identical factors.

        Returns:
            Tuple[complex, List[~.operation.Operator]: tuple containing the global phase and a list
            of the simplified factors
        """
    new_factors = _ProductFactorsGrouping()
    for factor in factors:
        simplified_factor = factor.simplify()
        new_factors.add(factor=simplified_factor)
    new_factors.remove_factors(wires=self.wires)
    return (new_factors.global_phase, new_factors.factors)
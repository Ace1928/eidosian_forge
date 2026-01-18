import warnings
import itertools
from copy import copy
from typing import List
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.qubit import Hamiltonian
from pennylane.queuing import QueuingManager
from .composite import CompositeOp
class _SumSummandsGrouping:
    """Utils class used for grouping sum summands together."""

    def __init__(self):
        self.queue = {}

    def add(self, summand: Operator, coeff=1, op_hash=None):
        """Add operator to the summands dictionary.

        If the operator hash is already in the dictionary, the coefficient is increased instead.

        Args:
            summand (Operator): operator to add to the summands dictionary
            coeff (int, optional): Coefficient of the operator. Defaults to 1.
            op_hash (int, optional): Hash of the operator. Defaults to None.
        """
        if isinstance(summand, qml.ops.SProd):
            coeff = summand.scalar if coeff == 1 else summand.scalar * coeff
            self.add(summand=summand.base, coeff=coeff)
        else:
            op_hash = summand.hash if op_hash is None else op_hash
            if op_hash in self.queue:
                self.queue[op_hash][0] += coeff
            else:
                self.queue[op_hash] = [copy(coeff), summand]

    def get_summands(self, cutoff=1e-12):
        """Get summands list.

        All summands with a coefficient less than cutoff are ignored.

        Args:
            cutoff (float, optional): Cutoff value. Defaults to 1.0e-12.
        """
        new_summands = []
        for coeff, summand in self.queue.values():
            if coeff == 1:
                new_summands.append(summand)
            elif abs(coeff) > cutoff:
                new_summands.append(qml.s_prod(coeff, summand))
        return new_summands
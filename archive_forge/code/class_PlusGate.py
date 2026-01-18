import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
class PlusGate(cirq.Gate):

    def __init__(self, dimension, increment=1):
        self.dimension = dimension
        self.increment = increment % dimension

    def _qid_shape_(self):
        return (self.dimension,)

    def _unitary_(self):
        inc = (self.increment - 1) % self.dimension + 1
        u = np.empty((self.dimension, self.dimension))
        u[inc:] = np.eye(self.dimension)[:-inc]
        u[:inc] = np.eye(self.dimension)[-inc:]
        return u
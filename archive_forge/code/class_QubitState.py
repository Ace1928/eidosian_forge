import math
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import log
from sympy.core.basic import _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import Matrix, zeros
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.state import Ket, Bra, State
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import (
from mpmath.libmp.libintmath import bitcount
class QubitState(State):
    """Base class for Qubit and QubitBra."""

    @classmethod
    def _eval_args(cls, args):
        if len(args) == 1 and isinstance(args[0], QubitState):
            return args[0].qubit_values
        if len(args) == 1 and isinstance(args[0], str):
            args = tuple((S.Zero if qb == '0' else S.One for qb in args[0]))
        else:
            args = tuple((S.Zero if qb == '0' else S.One if qb == '1' else qb for qb in args))
        args = tuple((_sympify(arg) for arg in args))
        for element in args:
            if element not in (S.Zero, S.One):
                raise ValueError('Qubit values must be 0 or 1, got: %r' % element)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        return ComplexSpace(2) ** len(args)

    @property
    def dimension(self):
        """The number of Qubits in the state."""
        return len(self.qubit_values)

    @property
    def nqubits(self):
        return self.dimension

    @property
    def qubit_values(self):
        """Returns the values of the qubits as a tuple."""
        return self.label

    def __len__(self):
        return self.dimension

    def __getitem__(self, bit):
        return self.qubit_values[int(self.dimension - bit - 1)]

    def flip(self, *bits):
        """Flip the bit(s) given."""
        newargs = list(self.qubit_values)
        for i in bits:
            bit = int(self.dimension - i - 1)
            if newargs[bit] == 1:
                newargs[bit] = 0
            else:
                newargs[bit] = 1
        return self.__class__(*tuple(newargs))
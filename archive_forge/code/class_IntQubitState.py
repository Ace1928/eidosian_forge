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
class IntQubitState(QubitState):
    """A base class for qubits that work with binary representations."""

    @classmethod
    def _eval_args(cls, args, nqubits=None):
        if len(args) == 1 and isinstance(args[0], QubitState):
            return QubitState._eval_args(args)
        elif not all((isinstance(a, (int, Integer)) for a in args)):
            raise ValueError('values must be integers, got (%s)' % (tuple((type(a) for a in args)),))
        if nqubits is not None:
            if not isinstance(nqubits, (int, Integer)):
                raise ValueError('nqubits must be an integer, got (%s)' % type(nqubits))
            if len(args) != 1:
                raise ValueError('too many positional arguments (%s). should be (number, nqubits=n)' % (args,))
            return cls._eval_args_with_nqubits(args[0], nqubits)
        if len(args) == 1 and args[0] > 1:
            rvalues = reversed(range(bitcount(abs(args[0]))))
            qubit_values = [args[0] >> i & 1 for i in rvalues]
            return QubitState._eval_args(qubit_values)
        elif len(args) == 2 and args[1] > 1:
            return cls._eval_args_with_nqubits(args[0], args[1])
        else:
            return QubitState._eval_args(args)

    @classmethod
    def _eval_args_with_nqubits(cls, number, nqubits):
        need = bitcount(abs(number))
        if nqubits < need:
            raise ValueError('cannot represent %s with %s bits' % (number, nqubits))
        qubit_values = [number >> i & 1 for i in reversed(range(nqubits))]
        return QubitState._eval_args(qubit_values)

    def as_int(self):
        """Return the numerical value of the qubit."""
        number = 0
        n = 1
        for i in reversed(self.qubit_values):
            number += n * i
            n = n << 1
        return number

    def _print_label(self, printer, *args):
        return str(self.as_int())

    def _print_label_pretty(self, printer, *args):
        label = self._print_label(printer, *args)
        return prettyForm(label)
    _print_label_repr = _print_label
    _print_label_latex = _print_label
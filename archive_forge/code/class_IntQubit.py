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
class IntQubit(IntQubitState, Qubit):
    """A qubit ket that store integers as binary numbers in qubit values.

    The differences between this class and ``Qubit`` are:

    * The form of the constructor.
    * The qubit values are printed as their corresponding integer, rather
      than the raw qubit values. The internal storage format of the qubit
      values in the same as ``Qubit``.

    Parameters
    ==========

    values : int, tuple
        If a single argument, the integer we want to represent in the qubit
        values. This integer will be represented using the fewest possible
        number of qubits.
        If a pair of integers and the second value is more than one, the first
        integer gives the integer to represent in binary form and the second
        integer gives the number of qubits to use.
        List of zeros and ones is also accepted to generate qubit by bit pattern.

    nqubits : int
        The integer that represents the number of qubits.
        This number should be passed with keyword ``nqubits=N``.
        You can use this in order to avoid ambiguity of Qubit-style tuple of bits.
        Please see the example below for more details.

    Examples
    ========

    Create a qubit for the integer 5:

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qubit import Qubit
        >>> q = IntQubit(5)
        >>> q
        |5>

    We can also create an ``IntQubit`` by passing a ``Qubit`` instance.

        >>> q = IntQubit(Qubit('101'))
        >>> q
        |5>
        >>> q.as_int()
        5
        >>> q.nqubits
        3
        >>> q.qubit_values
        (1, 0, 1)

    We can go back to the regular qubit form.

        >>> Qubit(q)
        |101>

    Please note that ``IntQubit`` also accepts a ``Qubit``-style list of bits.
    So, the code below yields qubits 3, not a single bit ``1``.

        >>> IntQubit(1, 1)
        |3>

    To avoid ambiguity, use ``nqubits`` parameter.
    Use of this keyword is recommended especially when you provide the values by variables.

        >>> IntQubit(1, nqubits=1)
        |1>
        >>> a = 1
        >>> IntQubit(a, nqubits=1)
        |1>
    """

    @classmethod
    def dual_class(self):
        return IntQubitBra

    def _eval_innerproduct_IntQubitBra(self, bra, **hints):
        return Qubit._eval_innerproduct_QubitBra(self, bra)
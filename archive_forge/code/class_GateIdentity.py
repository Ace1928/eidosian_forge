from collections import deque
from sympy.core.random import randint
from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
class GateIdentity(Basic):
    """Wrapper class for circuits that reduce to a scalar value.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    Parameters
    ==========

    args : Gate tuple
        A variable length tuple of Gates that form an identity.

    Examples
    ========

    Create a GateIdentity and look at its attributes:

    >>> from sympy.physics.quantum.identitysearch import GateIdentity
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> an_identity.circuit
    X(0)*Y(0)*Z(0)

    >>> an_identity.equivalent_ids
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}
    """

    def __new__(cls, *args):
        obj = Basic.__new__(cls, *args)
        obj._circuit = Mul(*args)
        obj._rules = generate_gate_rules(args)
        obj._eq_ids = generate_equivalent_ids(args)
        return obj

    @property
    def circuit(self):
        return self._circuit

    @property
    def gate_rules(self):
        return self._rules

    @property
    def equivalent_ids(self):
        return self._eq_ids

    @property
    def sequence(self):
        return self.args

    def __str__(self):
        """Returns the string of gates in a tuple."""
        return str(self.circuit)
import re
from copy import copy
from numbers import Number
from numpy import ndarray
import pennylane as qml
class FermiA(FermiWord):
    """FermiA(orbital)
    The fermionic annihilation operator :math:`a`

    For instance, the operator ``qml.FermiA(2)`` denotes :math:`a_2`. This operator applied
    to :math:`\\ket{0010}` gives :math:`\\ket{0000}`.

    Args:
        orbital(int): the non-negative integer indicating the orbital the operator acts on.

    .. note:: While the ``FermiA`` class represents a mathematical operator, it is not a PennyLane qubit :class:`~.Operator`.

    .. seealso:: :class:`~pennylane.FermiC`

    **Example**

    To construct the operator :math:`a_0`:

    >>> FermiA(0)
    a(0)

    This can be combined with the creation operator :class:`~pennylane.FermiC`. For example,
    :math:`a^{\\dagger}_0 a_1 a^{\\dagger}_2 a_3` can be constructed as:

    >>> qml.FermiC(0) * qml.FermiA(1) * qml.FermiC(2) * qml.FermiA(3)
    a⁺(0) a(1) a⁺(2) a(3)
    """

    def __init__(self, orbital):
        if not isinstance(orbital, int) or orbital < 0:
            raise ValueError(f'FermiA: expected a single, positive integer value for orbital, but received {orbital}')
        operator = {(0, orbital): '-'}
        super().__init__(operator)
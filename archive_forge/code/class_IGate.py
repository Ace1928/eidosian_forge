from typing import Optional
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array
@with_gate_array([[1, 0], [0, 1]])
class IGate(SingletonGate):
    """Identity gate.

    Identity gate corresponds to a single-qubit gate wait cycle,
    and should not be optimized or unrolled (it is an opaque gate).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.i` and
    :meth:`~qiskit.circuit.QuantumCircuit.id` methods.

    **Matrix Representation:**

    .. math::

        I = \\begin{pmatrix}
                1 & 0 \\\\
                0 & 1
            \\end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::
             ┌───┐
        q_0: ┤ I ├
             └───┘
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        """Create new Identity gate."""
        super().__init__('id', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def inverse(self, annotated: bool=False):
        """Returne the inverse gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            IGate: inverse gate (self-inverse).
        ."""
        return IGate()

    def power(self, exponent: float):
        """Raise gate to a power."""
        return IGate()

    def __eq__(self, other):
        return isinstance(other, IGate)
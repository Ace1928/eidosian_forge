import math
from math import pi
from typing import Optional
import numpy
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array
@with_gate_array([[1, 0], [0, (1 - 1j) / math.sqrt(2)]])
class TdgGate(SingletonGate):
    """Single qubit T-adjoint gate (~Z**0.25).

    It induces a :math:`-\\pi/4` phase.

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.tdg` method.

    **Matrix Representation:**

    .. math::

        Tdg = \\begin{pmatrix}
                1 & 0 \\\\
                0 & e^{-i\\pi/4}
            \\end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────┐
        q_0: ┤ Tdg ├
             └─────┘

    Equivalent to a :math:`-\\pi/4` radian rotation about the Z axis.
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        """Create new Tdg gate."""
        super().__init__('tdg', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate tdg a { u1(pi/4) a; }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(-pi / 4), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverse Tdg gate (i.e. T).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.TGate`.

        Returns:
            TGate: inverse of :class:`.TdgGate`
        """
        return TGate()

    def power(self, exponent: float):
        """Raise gate to a power."""
        return PhaseGate(-0.25 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, TdgGate)
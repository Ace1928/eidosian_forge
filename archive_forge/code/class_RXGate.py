import math
from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
class RXGate(Gate):
    """Single-qubit rotation about the X axis.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rx` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ Rx(ϴ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

        RX(\\theta) = \\exp\\left(-i \\rotationangle X\\right) =
            \\begin{pmatrix}
                \\cos\\left(\\rotationangle\\right)   & -i\\sin\\left(\\rotationangle\\right) \\\\
                -i\\sin\\left(\\rotationangle\\right) & \\cos\\left(\\rotationangle\\right)
            \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str]=None, *, duration=None, unit='dt'):
        """Create new RX gate."""
        super().__init__('rx', 1, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        gate rx(theta) a {r(theta, 0) a;}
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .r import RGate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RGate(self.params[0], 0), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, annotated: bool=False):
        """Return a (multi-)controlled-RX gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate can be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated and num_ctrl_qubits == 1:
            gate = CRXGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted RX gate.

        :math:`RX(\\lambda)^{\\dagger} = RX(-\\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RXGate` with an inverted parameter value.

        Returns:
            RXGate: inverse gate.
        """
        return RXGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the RX gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        theta, = self.params
        return RXGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RXGate):
            return self._compare_parameters(other)
        return False
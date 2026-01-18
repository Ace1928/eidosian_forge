import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
class U3Gate(Gate):
    """Generic single-qubit rotation gate with 3 Euler angles.

    .. warning::

       This gate is deprecated. Instead, the following replacements should be used

       .. math::

           U3(\\theta, \\phi, \\lambda) =  U(\\theta, \\phi, \\lambda)

       .. code-block:: python

          circuit = QuantumCircuit(1)
          circuit.u(theta, phi, lambda)

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤ U3(ϴ,φ,λ) ├
             └───────────┘

    **Matrix Representation:**

    .. math::

        \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

        U3(\\theta, \\phi, \\lambda) =
            \\begin{pmatrix}
                \\cos\\left(\\rotationangle\\right) & -e^{i\\lambda}\\sin\\left(\\rotationangle\\right) \\\\
                e^{i\\phi}\\sin\\left(\\rotationangle\\right) &
                e^{i(\\phi+\\lambda)}\\cos\\left(\\rotationangle\\right)
            \\end{pmatrix}

    .. note::

        The matrix representation shown here differs from the `OpenQASM 2.0 specification
        <https://doi.org/10.48550/arXiv.1707.03429>`_ by a global phase of
        :math:`e^{i(\\phi+\\lambda)/2}`.

    **Examples:**

    .. math::

        U3(\\theta, \\phi, \\lambda) = e^{-i \\frac{\\pi + \\theta}{2}} P(\\phi + \\pi) \\sqrt{X}
        P(\\theta + \\pi) \\sqrt{X} P(\\lambda)

    .. math::

        U3\\left(\\theta, -\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right) = RX(\\theta)

    .. math::

        U3(\\theta, 0, 0) = RY(\\theta)
    """

    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, lam: ParameterValueType, label: Optional[str]=None, *, duration=None, unit='dt'):
        """Create new U3 gate."""
        super().__init__('u3', 1, [theta, phi, lam], label=label, duration=duration, unit=unit)

    def inverse(self, annotated: bool=False):
        """Return inverted U3 gate.

        :math:`U3(\\theta,\\phi,\\lambda)^{\\dagger} =U3(-\\theta,-\\lambda,-\\phi))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.U3Gate` with inverse parameter values.

        Returns:
            U3Gate: inverse gate.
        """
        return U3Gate(-self.params[0], -self.params[2], -self.params[1])

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, annotated: bool=False):
        """Return a (multi-)controlled-U3 gate.

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
            gate = CU3Gate(*self.params, label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc.u(self.params[0], self.params[1], self.params[2], 0)
        self.definition = qc

    def __array__(self, dtype=complex):
        """Return a Numpy.array for the U3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = (float(theta), float(phi), float(lam))
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return numpy.array([[cos, -exp(1j * lam) * sin], [exp(1j * phi) * sin, exp(1j * (phi + lam)) * cos]], dtype=dtype)
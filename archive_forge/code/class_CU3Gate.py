import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
class CU3Gate(ControlledGate):
    """Controlled-U3 gate (3-parameter two-qubit gate).

    This is a controlled version of the U3 gate (generic single qubit rotation).
    It is restricted to 3 parameters, and so cannot cover generic two-qubit
    controlled gates).

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──────■──────
             ┌─────┴─────┐
        q_1: ┤ U3(ϴ,φ,λ) ├
             └───────────┘

    **Matrix representation:**

    .. math::

        \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

        CU3(\\theta, \\phi, \\lambda)\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| +
            U3(\\theta,\\phi,\\lambda) \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0                   & 0 & 0 \\\\
                0 & \\cos(\\rotationangle)           & 0 & -e^{i\\lambda}\\sin(\\rotationangle) \\\\
                0 & 0                   & 1 & 0 \\\\
                0 & e^{i\\phi}\\sin(\\rotationangle)  & 0 & e^{i(\\phi+\\lambda)}\\cos(\\rotationangle)
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────────┐
            q_0: ┤ U3(ϴ,φ,λ) ├
                 └─────┬─────┘
            q_1: ──────■──────

        .. math::

            \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

            CU3(\\theta, \\phi, \\lambda)\\ q_1, q_0 =
                |0\\rangle\\langle 0| \\otimes I +
                |1\\rangle\\langle 1| \\otimes U3(\\theta,\\phi,\\lambda) =
                \\begin{pmatrix}
                    1 & 0   & 0                  & 0 \\\\
                    0 & 1   & 0                  & 0 \\\\
                    0 & 0   & \\cos(\\rotationangle)          & -e^{i\\lambda}\\sin(\\rotationangle) \\\\
                    0 & 0   & e^{i\\phi}\\sin(\\rotationangle) & e^{i(\\phi+\\lambda)}\\cos(\\rotationangle)
                \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, lam: ParameterValueType, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CU3 gate."""
        super().__init__('cu3', 2, [theta, phi, lam], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=U3Gate(theta, phi, lam, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        """
        gate cu3(theta,phi,lambda) c, t
        { u1((lambda+phi)/2) c;
          u1((lambda-phi)/2) t;
          cx c,t;
          u3(-theta/2,0,-(phi+lambda)/2) t;
          cx c,t;
          u3(theta/2,phi,0) t;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate((self.params[2] + self.params[1]) / 2), [q[0]], []), (U1Gate((self.params[2] - self.params[1]) / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (U3Gate(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (U3Gate(self.params[0] / 2, self.params[1], 0), [q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverted CU3 gate.

        :math:`CU3(\\theta,\\phi,\\lambda)^{\\dagger} =CU3(-\\theta,-\\phi,-\\lambda))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CU3Gate` with inverse
                parameter values.

        Returns:
            CU3Gate: inverse gate.
        """
        return CU3Gate(-self.params[0], -self.params[2], -self.params[1], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=complex):
        """Return a numpy.array for the CU3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = (float(theta), float(phi), float(lam))
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, cos, 0, -exp(1j * lam) * sin], [0, 0, 1, 0], [0, exp(1j * phi) * sin, 0, exp(1j * (phi + lam)) * cos]], dtype=dtype)
        else:
            return numpy.array([[cos, 0, -exp(1j * lam) * sin, 0], [0, 1, 0, 0], [exp(1j * phi) * sin, 0, exp(1j * (phi + lam)) * cos, 0], [0, 0, 0, 1]], dtype=dtype)
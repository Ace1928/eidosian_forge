from cmath import exp
from typing import Optional, Union
from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
class CRZGate(ControlledGate):
    """Controlled-RZ gate.

    This is a diagonal but non-symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.crz` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Rz(λ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        CRZ(\\lambda)\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| + RZ(\\lambda) \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & e^{-i\\frac{\\lambda}{2}} & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & e^{i\\frac{\\lambda}{2}}
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Rz(λ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            CRZ(\\lambda)\\ q_1, q_0 =
                |0\\rangle\\langle 0| \\otimes I + |1\\rangle\\langle 1| \\otimes RZ(\\lambda) =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 \\\\
                    0 & 0 & e^{-i\\frac{\\lambda}{2}} & 0 \\\\
                    0 & 0 & 0 & e^{i\\frac{\\lambda}{2}}
                \\end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CU1Gate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CRZ gate."""
        super().__init__('crz', 2, [theta], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=RZGate(theta, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        """
        gate crz(lambda) a,b
        { rz(lambda/2) b; cx a,b;
          rz(-lambda/2) b; cx a,b;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RZGate(self.params[0] / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (RZGate(-self.params[0] / 2), [q[1]], []), (CXGate(), [q[0], q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverse CRZ gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CRZGate` with an inverted parameter value.

         Returns:
            CRZGate: inverse gate.
        """
        return CRZGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the CRZ gate."""
        import numpy
        arg = 1j * float(self.params[0]) / 2
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, exp(-arg), 0, 0], [0, 0, 1, 0], [0, 0, 0, exp(arg)]], dtype=dtype)
        else:
            return numpy.array([[exp(-arg), 0, 0, 0], [0, 1, 0, 0], [0, 0, exp(arg), 0], [0, 0, 0, 1]], dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, CRZGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False
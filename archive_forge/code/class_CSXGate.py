from math import pi
from typing import Optional, Union
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
@with_controlled_gate_array(_SX_ARRAY, num_ctrl_qubits=1)
class CSXGate(SingletonControlledGate):
    """Controlled-√X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.csx` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
             ┌─┴──┐
        q_1: ┤ √X ├
             └────┘

    **Matrix representation:**

    .. math::

        C\\sqrt{X} \\ q_0, q_1 =
        I \\otimes |0 \\rangle\\langle 0| + \\sqrt{X} \\otimes |1 \\rangle\\langle 1|  =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & (1 + i) / 2 & 0 & (1 - i) / 2 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & (1 - i) / 2 & 0 & (1 + i) / 2
            \\end{pmatrix}


    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be `q_1`. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌────┐
            q_0: ┤ √X ├
                 └─┬──┘
            q_1: ──■──

        .. math::

            C\\sqrt{X}\\ q_1, q_0 =
                |0 \\rangle\\langle 0| \\otimes I + |1 \\rangle\\langle 1| \\otimes \\sqrt{X} =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 \\\\
                    0 & 0 & (1 + i) / 2 & (1 - i) / 2 \\\\
                    0 & 0 & (1 - i) / 2 & (1 + i) / 2
                \\end{pmatrix}

    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CSX gate."""
        super().__init__('csx', 2, [], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=SXGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate csx a,b { h b; cu1(pi/2) a,b; h b; }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .u1 import CU1Gate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[1]], []), (CU1Gate(pi / 2), [q[0], q[1]], []), (HGate(), [q[1]], [])]
        for operation, qubits, clbits in rules:
            qc._append(operation, qubits, clbits)
        self.definition = qc

    def __eq__(self, other):
        return isinstance(other, CSXGate) and self.ctrl_state == other.ctrl_state
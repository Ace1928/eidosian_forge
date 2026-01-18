from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=1)
class CXGate(SingletonControlledGate):
    """Controlled-X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cx` and
    :meth:`~qiskit.circuit.QuantumCircuit.cnot` methods.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ X ├
             └───┘

    **Matrix representation:**

    .. math::

        CX\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle0| + X \\otimes |1\\rangle\\langle1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 1 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 1 & 0 & 0
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───┐
            q_0: ┤ X ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CX\\ q_1, q_0 =
                |0 \\rangle\\langle 0| \\otimes I + |1 \\rangle\\langle 1| \\otimes X =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 \\\\
                    0 & 0 & 0 & 1 \\\\
                    0 & 0 & 1 & 0
                \\end{pmatrix}


    In the computational basis, this gate flips the target qubit
    if the control qubit is in the :math:`|1\\rangle` state.
    In this sense it is similar to a classical XOR gate.

    .. math::
        `|a, b\\rangle \\rightarrow |a, a \\oplus b\\rangle`
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CX gate."""
        super().__init__('cx', 2, [], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=XGate(label=_base_label), _base_label=_base_label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, annotated: bool=False):
        """Return a controlled-X gate with more control lines.

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
        if not annotated:
            ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
            new_ctrl_state = self.ctrl_state << num_ctrl_qubits | ctrl_state
            gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 1, label=label, ctrl_state=new_ctrl_state, _base_label=self.label)
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted CX gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CXGate: inverse gate (self-inverse).
        """
        return CXGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, CXGate) and self.ctrl_state == other.ctrl_state
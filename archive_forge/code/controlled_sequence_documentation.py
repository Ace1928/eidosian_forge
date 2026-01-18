from copy import copy
import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires
from pennylane.ops.op_math.symbolicop import SymbolicOp
Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.CtrlSequence.decomposition`.

        Args:
            base (Operator): the operator that acts as the base for the sequence
            control_wires (Any or Iterable[Any]): the control wires for the sequence

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        .. code-block:: python

            dev = qml.device("default.qubit")
            op = qml.ControlledSequence(qml.RX(0.25, wires = 3), control = [0, 1, 2])

            @qml.qnode(dev)
            def circuit():
                op.decomposition()
                return qml.state()

        >>> print(qml.draw(circuit, wire_order=[0,1,2,3])())
        0: ─╭●─────────────────────────────────────┤  State
        1: ─│────────────╭●────────────────────────┤  State
        2: ─│────────────│────────────╭●───────────┤  State
        3: ─╰(RX(1.00))──╰(RX(0.50))──╰(RX(0.25))──┤  State

        To display the operators as powers of the base operator without further simplifcation,
        the `compute_decompostion` method can be used with `lazy=True`.

        .. code-block:: python

            dev = qml.device("default.qubit")
            op = qml.ControlledSequence(qml.RX(0.25, wires = 3), control = [0, 1, 2])

            @qml.qnode(dev)
            def circuit():
                op.compute_decomposition(base=op.base, control_wires=op.control, lazy=True)
                return qml.state()

        >>> print(qml.draw(circuit, wire_order=[0,1,2,3])())
        0: ─╭●─────────────────────────────────────┤  State
        1: ─│────────────╭●────────────────────────┤  State
        2: ─│────────────│────────────╭●───────────┤  State
        3: ─╰(RX(0.25))⁴─╰(RX(0.25))²─╰(RX(0.25))¹─┤  State

        
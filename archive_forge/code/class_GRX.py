import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
class GRX(GR):
    """Global RX gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────┐
        q_0: ┤0         ├
             │          │
        q_1: ┤1  GRX(ϴ) ├
             │          │
        q_2: ┤2         ├
             └──────────┘

    The global RX gate is native to atomic systems (ion traps, cold neutrals). The global RX
    can be applied to multiple qubits simultaneously.

    In the one-qubit case, this is equivalent to an RX(theta) operations,
    and is thus reduced to the RXGate. The global RX gate is a direct sum of RX
    operations on all individual qubits.

    .. math::

        GRX(\\theta) = \\exp(-i \\sum_{i=1}^{n} X_i \\theta/2)

    **Expanded Circuit:**

    .. plot::

        from qiskit.circuit.library import GRX
        from qiskit.visualization.library import _generate_circuit_library_visualization
        import numpy as np
        circuit = GRX(num_qubits=3, theta=np.pi/4)
        _generate_circuit_library_visualization(circuit)

    """

    def __init__(self, num_qubits: int, theta: float) -> None:
        """Create a new Global RX (GRX) gate.

        Args:
            num_qubits: number of qubits.
            theta: rotation angle about x-axis
        """
        super().__init__(num_qubits, theta, phi=0)
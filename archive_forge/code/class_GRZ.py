import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
class GRZ(QuantumCircuit):
    """Global RZ gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────┐
        q_0: ┤0         ├
             │          │
        q_1: ┤1  GRZ(φ) ├
             │          │
        q_2: ┤2         ├
             └──────────┘

    The global RZ gate is native to atomic systems (ion traps, cold neutrals). The global RZ
    can be applied to multiple qubits simultaneously.

    In the one-qubit case, this is equivalent to an RZ(phi) operation,
    and is thus reduced to the RZGate. The global RZ gate is a direct sum of RZ
    operations on all individual qubits.

    .. math::

        GRZ(\\phi) = \\exp(-i \\sum_{i=1}^{n} Z_i \\phi)

    **Expanded Circuit:**

    .. plot::

       from qiskit.circuit.library import GRZ
       from qiskit.visualization.library import _generate_circuit_library_visualization
       import numpy as np
       circuit = GRZ(num_qubits=3, phi=np.pi/2)
       _generate_circuit_library_visualization(circuit)

    """

    def __init__(self, num_qubits: int, phi: float) -> None:
        """Create a new Global RZ (GRZ) gate.

        Args:
            num_qubits: number of qubits.
            phi: rotation angle about z-axis
        """
        super().__init__(num_qubits, name='grz')
        self.rz(phi, self.qubits)
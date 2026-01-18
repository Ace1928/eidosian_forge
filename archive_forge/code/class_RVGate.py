import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.exceptions import CircuitError
class RVGate(Gate):
    """Rotation around arbitrary rotation axis :math:`v` where :math:`|v|` is
    angle of rotation in radians.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rv` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────────────────┐
        q_0: ┤ RV(v_x,v_y,v_z) ├
             └─────────────────┘

    **Matrix Representation:**

    .. math::

        \\newcommand{\\rotationangle}{|\\vec{v}|}
        \\newcommand{\\sinc}{\\text{sinc}}
            R(\\vec{v}) = e^{-i \\vec{v}\\cdot\\vec{\\sigma}} =
                \\begin{pmatrix}
                    \\cos\\left(\\rotationangle\\right) -i v_z \\sinc\\left(\\rotationangle\\right)
                    & -(i v_x + v_y) \\sinc\\left(\\rotationangle\\right) \\\\
                    -(i v_x - v_y) \\sinc\\left(\\rotationangle\\right)
                    & \\cos\\left(\\rotationangle\\right) + i v_z \\sinc\\left(\\rotationangle\\right)
                \\end{pmatrix}
    """

    def __init__(self, v_x, v_y, v_z, basis='U'):
        """Create new rv single-qubit gate.

        Args:
            v_x (float): x-component
            v_y (float): y-component
            v_z (float): z-component
            basis (str, optional): basis (see
                :class:`~qiskit.synthesis.one_qubit.one_qubit_decompose.OneQubitEulerDecomposer`)
        """
        from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
        super().__init__('rv', 1, [v_x, v_y, v_z])
        self._decomposer = OneQubitEulerDecomposer(basis=basis)

    def _define(self):
        try:
            self.definition = self._decomposer(self.to_matrix())
        except TypeError as ex:
            raise CircuitError(f'The {self.name} gate cannot be decomposed with unbound parameters') from ex

    def inverse(self, annotated: bool=False):
        """Invert this gate."""
        vx, vy, vz = self.params
        return RVGate(-vx, -vy, -vz)

    def to_matrix(self):
        """Return a numpy.array for the R(v) gate."""
        v = numpy.asarray(self.params, dtype=float)
        angle = numpy.sqrt(v.dot(v))
        if angle == 0:
            return numpy.array([[1, 0], [0, 1]])
        nx, ny, nz = v / angle
        sin = numpy.sin(angle / 2)
        cos = numpy.cos(angle / 2)
        return numpy.array([[cos - 1j * nz * sin, (-ny - 1j * nx) * sin], [(ny - 1j * nx) * sin, cos + 1j * nz * sin]])
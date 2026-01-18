import numpy as np
import cirq
from cirq._doc import document
class SycamoreGate(cirq.FSimGate):
    """The Sycamore gate is a two-qubit gate equivalent to FSimGate(π/2, π/6).

    The unitary of this gate is

        [[1, 0, 0, 0],
         [0, 0, -1j, 0],
         [0, -1j, 0, 0],
         [0, 0, 0, exp(- 1j * π/6)]]

    This gate can be performed on the Google's Sycamore chip and
    is close to the gates that were used to demonstrate beyond
    classical resuts used in this paper:
    https://www.nature.com/articles/s41586-019-1666-5
    """

    def __init__(self):
        super().__init__(theta=np.pi / 2, phi=np.pi / 6)

    def __repr__(self) -> str:
        return 'cirq_google.SYC'

    def __str__(self) -> str:
        return 'SYC'

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return ('SYC', 'SYC')

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, [])
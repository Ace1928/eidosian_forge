from typing import Any, Dict, Sequence, Union
import cmath
import math
import cirq
from cirq import protocols
from cirq._doc import document
import numpy as np
@cirq.value.value_equality
class GPI2Gate(cirq.Gate):
    """The GPI2 gate is a single qubit gate representing a pi/2 pulse.

    The unitary matrix of this gate is
    $$
    \\frac{1}{\\sqrt{2}}
    \\begin{bmatrix}
        1 & -i e^{-i \\phi} \\\\
        -i e^{-i \\phi} & 1
    \\end{bmatrix}
    $$

    See [IonQ best practices](https://ionq.com/docs/getting-started-with-native-gates){:external}.
    """

    def __init__(self, *, phi):
        self.phi = phi

    def _unitary_(self) -> np.ndarray:
        top = -1j * cmath.exp(self.phase * 2 * math.pi * -1j)
        bot = -1j * cmath.exp(self.phase * 2 * math.pi * 1j)
        return np.array([[1, top], [bot, 1]]) / math.sqrt(2)

    @property
    def phase(self) -> float:
        return self.phi

    def __str__(self) -> str:
        return 'GPI2'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Union[str, 'protocols.CircuitDiagramInfo']:
        return protocols.CircuitDiagramInfo(wire_symbols=(f'GPI2({self.phase!r})',))

    def _num_qubits_(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f'cirq_ionq.GPI2Gate(phi={self.phi!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['phi'])

    def _value_equality_values_(self) -> Any:
        return self.phi

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return GPI2Gate(phi=self.phi + 0.5)
        return NotImplemented
from typing import (
import numpy as np
import sympy
import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import controlled_gate, eigen_gate, gate_features, raw_types, control_values as cv
from cirq.type_workarounds import NotImplementedType
from cirq.ops.swap_gates import ISWAP, SWAP, ISwapPowGate, SwapPowGate
from cirq.ops.measurement_gate import MeasurementGate
imports.
class Rx(XPowGate):
    """A gate with matrix $e^{-i X t/2}$ that rotates around the X axis of the Bloch sphere by $t$.

    The unitary matrix of `cirq.Rx(rads=t)` is:
    $$
    e^{-i X t /2} =
        \\begin{bmatrix}
            \\cos(t/2) & -i \\sin(t/2) \\\\
            -i \\sin(t/2) & \\cos(t/2)
        \\end{bmatrix}
    $$

    This gate corresponds to the traditionally defined rotation matrices about the Pauli X axis.
    """

    def __init__(self, *, rads: value.TParamVal):
        """Initialize an Rx (`cirq.XPowGate`).

        Args:
            rads: Radians to rotate about the X axis of the Bloch sphere.
        """
        self._rads = rads
        super().__init__(exponent=rads / _pi(rads), global_shift=-0.5)

    def _with_exponent(self, exponent: value.TParamVal) -> 'Rx':
        return Rx(rads=exponent * _pi(exponent))

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Union[str, 'protocols.CircuitDiagramInfo']:
        angle_str = self._format_exponent_as_angle(args)
        return f'Rx({angle_str})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Rx(π)'
        return f'Rx({self._exponent}π)'

    def __repr__(self) -> str:
        return f'cirq.Rx(rads={proper_repr(self._rads)})'

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        return args.format('rx({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def _json_dict_(self) -> Dict[str, Any]:
        return {'rads': self._rads}

    @classmethod
    def _from_json_dict_(cls, rads, **kwargs) -> 'Rx':
        return cls(rads=rads)
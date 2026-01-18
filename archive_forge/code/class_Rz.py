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
class Rz(ZPowGate):
    """A gate with matrix $e^{-i Z t/2}$ that rotates around the Z axis of the Bloch sphere by $t$.

    The unitary matrix of `cirq.Rz(rads=t)` is:
    $$
    e^{-i Z t /2} =
        \\begin{bmatrix}
            e^{-it/2} & 0 \\\\
            0 & e^{it/2}
        \\end{bmatrix}
    $$

    This gate corresponds to the traditionally defined rotation matrices about the Pauli Z axis.
    """

    def __init__(self, *, rads: value.TParamVal):
        """Initialize an Rz (`cirq.ZPowGate`).

        Args:
            rads: Radians to rotate about the Z axis of the Bloch sphere.
        """
        self._rads = rads
        super().__init__(exponent=rads / _pi(rads), global_shift=-0.5)

    def _with_exponent(self, exponent: value.TParamVal) -> 'Rz':
        return Rz(rads=exponent * _pi(exponent))

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Union[str, 'protocols.CircuitDiagramInfo']:
        angle_str = self._format_exponent_as_angle(args)
        return f'Rz({angle_str})'

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Rz(π)'
        return f'Rz({self._exponent}π)'

    def __repr__(self) -> str:
        return f'cirq.Rz(rads={proper_repr(self._rads)})'

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        return args.format('rz({0:half_turns}) {1};\n', self._exponent, qubits[0])

    def _json_dict_(self) -> Dict[str, Any]:
        return {'rads': self._rads}

    @classmethod
    def _from_json_dict_(cls, rads, **kwargs) -> 'Rz':
        return cls(rads=rads)
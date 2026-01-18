from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Sequence
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
class MSGate(XXPowGate):
    """The Mølmer–Sørensen gate, a native two-qubit operation in ion traps.

    A rotation around the XX axis in the two-qubit bloch sphere.

    The gate implements the following unitary:

        exp(-i t XX) = [ cos(t)   0        0       -isin(t)]
                       [ 0        cos(t)  -isin(t)  0      ]
                       [ 0       -isin(t)  cos(t)   0      ]
                       [-isin(t)  0        0        cos(t) ]
    """

    def __init__(self, *, rads: float):
        XXPowGate.__init__(self, exponent=rads * 2 / np.pi, global_shift=-0.5)
        self.rads = rads

    def _with_exponent(self, exponent: value.TParamVal) -> Self:
        return type(self)(rads=exponent * np.pi / 2)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Union[str, 'protocols.CircuitDiagramInfo']:
        angle_str = self._format_exponent_as_angle(args, order=4)
        symbol = f'MS({angle_str})'
        return protocols.CircuitDiagramInfo(wire_symbols=(symbol, symbol))

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'MS(π/2)'
        return f'MS({self._exponent!r}π/2)'

    def __repr__(self) -> str:
        if self._exponent == 1:
            return 'cirq.ms(np.pi/2)'
        return f'cirq.ms({self._exponent!r}*np.pi/2)'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['rads'])

    @classmethod
    def _from_json_dict_(cls, rads: float, **kwargs: Any) -> 'MSGate':
        return cls(rads=rads)
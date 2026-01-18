import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def _wire_symbols_including_formatted_exponent(self, args: 'cirq.CircuitDiagramInfoArgs', *, preferred_exponent_index: Optional[int]=None) -> List[str]:
    result = list(self.wire_symbols)
    exponent = self._formatted_exponent(args)
    if exponent is not None:
        ks: Sequence[int]
        if self.exponent_qubit_index is not None:
            ks = (self.exponent_qubit_index,)
        elif not self.connected:
            ks = range(len(result))
        elif preferred_exponent_index is not None:
            ks = (preferred_exponent_index,)
        else:
            ks = (0,)
        for k in ks:
            result[k] += f'^{exponent}'
    return result
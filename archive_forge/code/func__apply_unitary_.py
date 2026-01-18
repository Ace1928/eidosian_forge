from typing import Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
    return cirq.apply_unitary(self.target_gate.controlled(control_values=self.cvs), args)
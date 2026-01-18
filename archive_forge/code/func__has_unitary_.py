from typing import Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
def _has_unitary_(self) -> bool:
    return True
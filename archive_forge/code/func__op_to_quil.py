import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _op_to_quil(self, op: cirq.Operation) -> str:
    quil_str = self._op_to_maybe_quil(op)
    if not quil_str:
        raise ValueError("Can't convert Operation to string")
    return quil_str
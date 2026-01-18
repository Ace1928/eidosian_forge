import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _near_mod_n(self, e: Union[float, sympy.Expr], t: float, n: float) -> bool:
    """Returns whether a value, e, translated by t, is equal to 0 mod n.

        Note that, despite the typing, e should actually always be a float
        since the gate is checked for parameterization before this point.
        """
    return abs((cast(float, e) - t + 1) % n - 1) <= self.atol
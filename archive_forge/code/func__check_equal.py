from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union
import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str
def _check_equal(self, g1: POSSIBLE_FSIM_GATES, g2: POSSIBLE_FSIM_GATES) -> bool:
    if not self.allow_symbols:
        return g1 == g2 and (not (cirq.is_parameterized(g1) or cirq.is_parameterized(g2)))
    if self._get_value_equality_values_cls(g1) != self._get_value_equality_values_cls(g2):
        return False
    return self._approx_eq_or_symbol(self._get_value_equality_values(g1), self._get_value_equality_values(g2))
import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
def _commutes_on_qids_(self, qids: 'Sequence[cirq.Qid]', other: Any, *, atol: float=1e-08) -> Union[bool, NotImplementedType, None]:
    return NotImplemented
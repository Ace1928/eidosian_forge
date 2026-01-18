from typing import Any, overload, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def _strat_commutes_from_commutes(v1: Any, v2: Any, *, atol: Union[int, float]=1e-08) -> Union[bool, NotImplementedType, None]:
    """Attempts to determine commutativity via the objects' _commutes_
    method."""
    for a, b in [(v1, v2), (v2, v1)]:
        getter = getattr(a, '_commutes_', None)
        if getter is None:
            continue
        val = getter(b, atol=atol)
        if val is not NotImplemented:
            return val
    return NotImplemented
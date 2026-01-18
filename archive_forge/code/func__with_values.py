import dataclasses
from typing import Any, Dict, List, Sequence, Set, Type, TypeVar, Union
import numpy as np
import cirq, cirq_google
from cirq import _compat, devices
from cirq.devices import noise_utils
from cirq.transformers.heuristic_decompositions import gate_tabulation_math_utils
def _with_values(original: Dict[T, V], val: Union[V, Dict[T, V]]) -> Dict[T, V]:
    """Returns a copy of `original` using values from `val`.

    If val is a single value, all keys are mapped to that value. If val is a
    dict, the union of original and val is returned, using values from val for
    any conflicting keys.
    """
    if isinstance(val, dict):
        return {**original, **val}
    return {k: val for k in original}
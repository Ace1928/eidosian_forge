import numbers
from typing import Any, cast, Dict, Iterator, Mapping, Optional, TYPE_CHECKING, Union
import numpy as np
import sympy
from sympy.core import numbers as sympy_numbers
from cirq._compat import proper_repr
from cirq._doc import document
def _value_of_recursive(self, value: 'cirq.TParamKey') -> 'cirq.TParamValComplex':
    if value in self._deep_eval_map:
        v = self._deep_eval_map[value]
        if v is _RECURSION_FLAG:
            raise RecursionError('Evaluation of {value} indirectly contains itself.')
        return v
    self._deep_eval_map[value] = _RECURSION_FLAG
    v = self.value_of(value, recursive=False)
    if v == value:
        self._deep_eval_map[value] = v
    else:
        self._deep_eval_map[value] = self.value_of(v, recursive=True)
    return self._deep_eval_map[value]
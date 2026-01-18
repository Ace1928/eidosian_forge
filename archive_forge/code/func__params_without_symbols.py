from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
def _params_without_symbols(resolver: resolver.ParamResolver) -> Params:
    for sym, val in resolver.param_dict.items():
        if isinstance(sym, sympy.Symbol):
            sym = sym.name
        yield (cast(str, sym), cast(float, val))
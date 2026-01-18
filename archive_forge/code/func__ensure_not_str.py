from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import numbers
import sympy
from cirq import protocols
from cirq.study import resolver, sweeps, sweepable
def _ensure_not_str(param: Union[sympy.Expr, 'cirq.TParamValComplex', str]) -> Union[sympy.Expr, 'cirq.TParamValComplex']:
    if isinstance(param, str):
        return sympy.Symbol(param)
    return param
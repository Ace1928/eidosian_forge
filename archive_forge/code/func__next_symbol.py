from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import numbers
import sympy
from cirq import protocols
from cirq.study import resolver, sweeps, sweepable
def _next_symbol(self, val: sympy.Expr) -> sympy.Symbol:
    name = self.get_param_name(val)
    symbol = sympy.Symbol(name)
    collision = 0
    while symbol in self._taken_symbols:
        collision += 1
        symbol = sympy.Symbol(f'{name}_{collision}')
    return symbol
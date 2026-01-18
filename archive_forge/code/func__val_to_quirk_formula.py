from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def _val_to_quirk_formula(t: Union[float, sympy.Basic]) -> str:
    if isinstance(t, sympy.Basic):
        if not set(t.free_symbols) <= {sympy.Symbol('t')}:
            raise ValueError(f'Symbol other than "t": {t!r}.')
        if not _is_supported_formula(t):
            raise ValueError(f'Formula uses unsupported operations: {t!r}')
        return str(t)
    return f'{float(t):.4f}'
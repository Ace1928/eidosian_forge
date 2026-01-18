from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def _is_supported_formula(formula: sympy.Basic) -> bool:
    if isinstance(formula, (sympy.Symbol, sympy.Integer, sympy.Float, sympy.Rational, sympy.NumberSymbol)):
        return True
    if isinstance(formula, (sympy.Add, sympy.Mul)):
        return all((_is_supported_formula(f) for f in formula.args))
    return False
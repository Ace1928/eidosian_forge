import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
def _check_and_get_real_coef(observable: 'cirq.PauliString', atol: float):
    """Assert that a PauliString has a real coefficient and return it."""
    coef = observable.coefficient
    if isinstance(coef, sympy.Expr) or not np.isclose(coef.imag, 0, atol=atol):
        raise ValueError(f'{observable} has a complex or symbolic coefficient.')
    return coef.real
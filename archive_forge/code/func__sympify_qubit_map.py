from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
def _sympify_qubit_map(mapping):
    new_map = {}
    for key in mapping:
        new_map[key] = sympify(mapping[key])
    return new_map
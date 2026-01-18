from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
Insert a circuit into another quantum circuit.

    Explanation
    ===========

    random_insert randomly chooses a location in the circuit to insert
    a randomly selected circuit from amongst the given choices.

    Parameters
    ==========

    circuit : Gate tuple or Mul
        A tuple or Mul of Gates representing a quantum circuit
    choices : list
        Set of circuit choices
    seed : int or list
        seed used for _randrange; to override the random selections, give
        a list two integers, [i, j] where i is the circuit location where
        choice[j] will be inserted.

    Notes
    =====

    Indices for insertion should be [0, n] if n is the length of the
    circuit.
    
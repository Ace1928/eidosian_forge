from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _check_free_space_horizontal(dom_i, dom_j, cod_j, grid):
    """
        For a horizontal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """
    if dom_j < cod_j:
        start, end = (dom_j, cod_j)
        backwards = False
    else:
        start, end = (cod_j, dom_j)
        backwards = True
    if dom_i == 0:
        free_up = True
    else:
        free_up = all((grid[dom_i - 1, j] for j in range(start, end + 1)))
    if dom_i == grid.height - 1:
        free_down = True
    else:
        free_down = not any((grid[dom_i + 1, j] for j in range(start, end + 1)))
    return (free_up, free_down, backwards)
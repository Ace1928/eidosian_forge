from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _empty_point(pt, grid):
    """
        Checks if the cell at coordinates ``pt`` is either empty or
        out of the bounds of the grid.
        """
    if pt[0] < 0 or pt[1] < 0 or pt[0] >= grid.height or (pt[1] >= grid.width):
        return True
    return grid[pt] is None
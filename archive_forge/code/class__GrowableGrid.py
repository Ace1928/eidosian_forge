from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
class _GrowableGrid:
    """
    Holds a growable grid of objects.

    Explanation
    ===========

    It is possible to append or prepend a row or a column to the grid
    using the corresponding methods.  Prepending rows or columns has
    the effect of changing the coordinates of the already existing
    elements.

    This class currently represents a naive implementation of the
    functionality with little attempt at optimisation.
    """

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._array = [[None for j in range(width)] for i in range(height)]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __getitem__(self, i_j):
        """
        Returns the element located at in the i-th line and j-th
        column.
        """
        i, j = i_j
        return self._array[i][j]

    def __setitem__(self, i_j, newvalue):
        """
        Sets the element located at in the i-th line and j-th
        column.
        """
        i, j = i_j
        self._array[i][j] = newvalue

    def append_row(self):
        """
        Appends an empty row to the grid.
        """
        self._height += 1
        self._array.append([None for j in range(self._width)])

    def append_column(self):
        """
        Appends an empty column to the grid.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].append(None)

    def prepend_row(self):
        """
        Prepends the grid with an empty row.
        """
        self._height += 1
        self._array.insert(0, [None for j in range(self._width)])

    def prepend_column(self):
        """
        Prepends the grid with an empty column.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].insert(0, None)
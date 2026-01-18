from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _grow_pseudopod(triangles, fringe, grid, skeleton, placed_objects):
    """
        Starting from an object in the existing structure on the ``grid``,
        adds an edge to which a triangle from ``triangles`` could be
        welded.  If this method has found a way to do so, it returns
        the object it has just added.

        This method should be applied when ``_weld_triangle`` cannot
        find weldings any more.
        """
    for i in range(grid.height):
        for j in range(grid.width):
            obj = grid[i, j]
            if not obj:
                continue

            def good_triangle(tri):
                objs = DiagramGrid._triangle_objects(tri)
                return obj in objs and placed_objects & objs - {obj} == set()
            tris = [tri for tri in triangles if good_triangle(tri)]
            if not tris:
                continue
            tri = tris[0]
            candidates = sorted([e for e in tri if skeleton[e]], key=lambda e: FiniteSet(*e).sort_key())
            edges = [e for e in candidates if obj in e]
            edge = edges[0]
            other_obj = tuple(edge - frozenset([obj]))[0]
            neighbours = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]
            for pt in neighbours:
                if DiagramGrid._empty_point(pt, grid):
                    offset = DiagramGrid._put_object(pt, other_obj, grid, fringe)
                    i += offset[0]
                    j += offset[1]
                    pt = (pt[0] + offset[0], pt[1] + offset[1])
                    fringe.append(((i, j), pt))
                    return other_obj
    return None
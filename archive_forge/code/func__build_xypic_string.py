from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _build_xypic_string(diagram, grid, morphisms, morphisms_str_info, diagram_format):
    """
        Given a collection of :class:`ArrowStringDescription`
        describing the morphisms of a diagram and the object layout
        information of a diagram, produces the final Xy-pic picture.
        """
    object_morphisms = {}
    for obj in diagram.objects:
        object_morphisms[obj] = []
    for morphism in morphisms:
        object_morphisms[morphism.domain].append(morphism)
    result = '\\xymatrix%s{\n' % diagram_format
    for i in range(grid.height):
        for j in range(grid.width):
            obj = grid[i, j]
            if obj:
                result += latex(obj) + ' '
                morphisms_to_draw = object_morphisms[obj]
                for morphism in morphisms_to_draw:
                    result += str(morphisms_str_info[morphism]) + ' '
            if j < grid.width - 1:
                result += '& '
        if i < grid.height - 1:
            result += '\\\\'
        result += '\n'
    result += '}\n'
    return result
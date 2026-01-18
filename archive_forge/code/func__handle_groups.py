from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _handle_groups(diagram, groups, merged_morphisms, hints):
    """
        Given the slightly preprocessed morphisms of the diagram,
        produces a grid laid out according to ``groups``.

        If a group has hints, it is laid out with those hints only,
        without any influence from ``hints``.  Otherwise, it is laid
        out with ``hints``.
        """

    def lay_out_group(group, local_hints):
        """
            If ``group`` is a set of objects, uses a ``DiagramGrid``
            to lay it out and returns the grid.  Otherwise returns the
            object (i.e., ``group``).  If ``local_hints`` is not
            empty, it is supplied to ``DiagramGrid`` as the dictionary
            of hints.  Otherwise, the ``hints`` argument of
            ``_handle_groups`` is used.
            """
        if isinstance(group, FiniteSet):
            for obj in group:
                obj_groups[obj] = group
            if local_hints:
                groups_grids[group] = DiagramGrid(diagram.subdiagram_from_objects(group), **local_hints)
            else:
                groups_grids[group] = DiagramGrid(diagram.subdiagram_from_objects(group), **hints)
        else:
            obj_groups[group] = group

    def group_to_finiteset(group):
        """
            Converts ``group`` to a :class:``FiniteSet`` if it is an
            iterable.
            """
        if iterable(group):
            return FiniteSet(*group)
        else:
            return group
    obj_groups = {}
    groups_grids = {}
    if isinstance(groups, (dict, Dict)):
        finiteset_groups = {}
        for group, local_hints in groups.items():
            finiteset_group = group_to_finiteset(group)
            finiteset_groups[finiteset_group] = local_hints
            lay_out_group(group, local_hints)
        groups = finiteset_groups
    else:
        finiteset_groups = []
        for group in groups:
            finiteset_group = group_to_finiteset(group)
            finiteset_groups.append(finiteset_group)
            lay_out_group(finiteset_group, None)
        groups = finiteset_groups
    new_morphisms = []
    for morphism in merged_morphisms:
        dom = obj_groups[morphism.domain]
        cod = obj_groups[morphism.codomain]
        if dom != cod:
            new_morphisms.append(NamedMorphism(dom, cod, 'dummy'))
    top_grid = DiagramGrid(Diagram(new_morphisms))

    def group_size(group):
        """
            For the supplied group (or object, eventually), returns
            the size of the cell that will hold this group (object).
            """
        if group in groups_grids:
            grid = groups_grids[group]
            return (grid.height, grid.width)
        else:
            return (1, 1)
    row_heights = [max((group_size(top_grid[i, j])[0] for j in range(top_grid.width))) for i in range(top_grid.height)]
    column_widths = [max((group_size(top_grid[i, j])[1] for i in range(top_grid.height))) for j in range(top_grid.width)]
    grid = _GrowableGrid(sum(column_widths), sum(row_heights))
    real_row = 0
    real_column = 0
    for logical_row in range(top_grid.height):
        for logical_column in range(top_grid.width):
            obj = top_grid[logical_row, logical_column]
            if obj in groups_grids:
                local_grid = groups_grids[obj]
                for i in range(local_grid.height):
                    for j in range(local_grid.width):
                        grid[real_row + i, real_column + j] = local_grid[i, j]
            else:
                grid[real_row, real_column] = obj
            real_column += column_widths[logical_column]
        real_column = 0
        real_row += row_heights[logical_row]
    return grid
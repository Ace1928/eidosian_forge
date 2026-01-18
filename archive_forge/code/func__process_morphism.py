from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def _process_morphism(self, diagram, grid, morphism, object_coords, morphisms, morphisms_str_info):
    """
        Given the required information, produces the string
        representation of ``morphism``.
        """

    def repeat_string_cond(times, str_gt, str_lt):
        """
            If ``times > 0``, repeats ``str_gt`` ``times`` times.
            Otherwise, repeats ``str_lt`` ``-times`` times.
            """
        if times > 0:
            return str_gt * times
        else:
            return str_lt * -times

    def count_morphisms_undirected(A, B):
        """
            Counts how many processed morphisms there are between the
            two supplied objects.
            """
        return len([m for m in morphisms_str_info if {m.domain, m.codomain} == {A, B}])

    def count_morphisms_filtered(dom, cod, curving):
        """
            Counts the processed morphisms which go out of ``dom``
            into ``cod`` with curving ``curving``.
            """
        return len([m for m, m_str_info in morphisms_str_info.items() if (m.domain, m.codomain) == (dom, cod) and m_str_info.curving == curving])
    i, j = object_coords[morphism.domain]
    target_i, target_j = object_coords[morphism.codomain]
    delta_i = target_i - i
    delta_j = target_j - j
    vertical_direction = repeat_string_cond(delta_i, 'd', 'u')
    horizontal_direction = repeat_string_cond(delta_j, 'r', 'l')
    curving = ''
    label_pos = '^'
    looping_start = ''
    looping_end = ''
    if delta_i == 0 and delta_j == 0:
        curving, label_pos, looping_start, looping_end = XypicDiagramDrawer._process_loop_morphism(i, j, grid, morphisms_str_info, object_coords)
    elif delta_i == 0 and abs(j - target_j) > 1:
        curving, label_pos = XypicDiagramDrawer._process_horizontal_morphism(i, j, target_j, grid, morphisms_str_info, object_coords)
    elif delta_j == 0 and abs(i - target_i) > 1:
        curving, label_pos = XypicDiagramDrawer._process_vertical_morphism(i, j, target_i, grid, morphisms_str_info, object_coords)
    count = count_morphisms_undirected(morphism.domain, morphism.codomain)
    curving_amount = ''
    if curving:
        curving_amount = self.default_curving_amount + count * self.default_curving_step
    elif count:
        curving = '^'
        filtered_morphisms = count_morphisms_filtered(morphism.domain, morphism.codomain, curving)
        curving_amount = self.default_curving_amount + filtered_morphisms * self.default_curving_step
    morphism_name = ''
    if isinstance(morphism, IdentityMorphism):
        morphism_name = 'id_{%s}' + latex(grid[i, j])
    elif isinstance(morphism, CompositeMorphism):
        component_names = [latex(Symbol(component.name)) for component in morphism.components]
        component_names.reverse()
        morphism_name = '\\circ '.join(component_names)
    elif isinstance(morphism, NamedMorphism):
        morphism_name = latex(Symbol(morphism.name))
    return ArrowStringDescription(self.unit, curving, curving_amount, looping_start, looping_end, horizontal_direction, vertical_direction, label_pos, morphism_name)
from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _process_loop_morphism(i, j, grid, morphisms_str_info, object_coords):
    """
        Produces the information required for constructing the string
        representation of a loop morphism.  This function is invoked
        from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
    curving = ''
    label_pos = '^'
    looping_start = ''
    looping_end = ''
    quadrant = [0, 0, 0, 0]
    obj = grid[i, j]
    for m, m_str_info in morphisms_str_info.items():
        if m.domain == obj and m.codomain == obj:
            l_s, l_e = (m_str_info.looping_start, m_str_info.looping_end)
            if (l_s, l_e) == ('r', 'u'):
                quadrant[0] += 1
            elif (l_s, l_e) == ('u', 'l'):
                quadrant[1] += 1
            elif (l_s, l_e) == ('l', 'd'):
                quadrant[2] += 1
            elif (l_s, l_e) == ('d', 'r'):
                quadrant[3] += 1
            continue
        if m.domain == obj:
            end_i, end_j = object_coords[m.codomain]
            goes_out = True
        elif m.codomain == obj:
            end_i, end_j = object_coords[m.domain]
            goes_out = False
        else:
            continue
        d_i = end_i - i
        d_j = end_j - j
        m_curving = m_str_info.curving
        if d_i != 0 and d_j != 0:
            if d_i > 0 and d_j > 0:
                quadrant[0] += 1
            elif d_i > 0 and d_j < 0:
                quadrant[1] += 1
            elif d_i < 0 and d_j < 0:
                quadrant[2] += 1
            elif d_i < 0 and d_j > 0:
                quadrant[3] += 1
        elif d_i == 0:
            if d_j > 0:
                if goes_out:
                    upper_quadrant = 0
                    lower_quadrant = 3
                else:
                    upper_quadrant = 3
                    lower_quadrant = 0
            elif goes_out:
                upper_quadrant = 2
                lower_quadrant = 1
            else:
                upper_quadrant = 1
                lower_quadrant = 2
            if m_curving:
                if m_curving == '^':
                    quadrant[upper_quadrant] += 1
                elif m_curving == '_':
                    quadrant[lower_quadrant] += 1
            else:
                quadrant[upper_quadrant] += 1
                quadrant[lower_quadrant] += 1
        elif d_j == 0:
            if d_i < 0:
                if goes_out:
                    left_quadrant = 1
                    right_quadrant = 0
                else:
                    left_quadrant = 0
                    right_quadrant = 1
            elif goes_out:
                left_quadrant = 3
                right_quadrant = 2
            else:
                left_quadrant = 2
                right_quadrant = 3
            if m_curving:
                if m_curving == '^':
                    quadrant[left_quadrant] += 1
                elif m_curving == '_':
                    quadrant[right_quadrant] += 1
            else:
                quadrant[left_quadrant] += 1
                quadrant[right_quadrant] += 1
    freest_quadrant = 0
    for i in range(4):
        if quadrant[i] < quadrant[freest_quadrant]:
            freest_quadrant = i
    looping_start, looping_end = [('r', 'u'), ('u', 'l'), ('l', 'd'), ('d', 'r')][freest_quadrant]
    return (curving, label_pos, looping_start, looping_end)
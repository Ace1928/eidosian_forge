from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
def _cycle_permute(l):
    """ Cyclic permutations based on canonical ordering

    Explanation
    ===========

    This method does the sort based ascii values while
    a better approach would be to used lexicographic sort.

    TODO: Handle condition such as symbols have subscripts/superscripts
    in case of lexicographic sort

    """
    if len(l) == 1:
        return l
    min_item = min(l, key=default_sort_key)
    indices = [i for i, x in enumerate(l) if x == min_item]
    le = list(l)
    le.extend(l)
    indices.append(len(l) + indices[0])
    sublist = [[le[indices[i]:indices[i + 1]]] for i in range(len(indices) - 1)]
    idx = sublist.index(min(sublist))
    ordered_l = le[indices[idx]:indices[idx] + len(l)]
    return ordered_l
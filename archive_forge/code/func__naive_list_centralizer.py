from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base
def _naive_list_centralizer(self, other, af=False):
    from sympy.combinatorics.perm_groups import PermutationGroup
    '\n    Return a list of elements for the centralizer of a subgroup/set/element.\n\n    Explanation\n    ===========\n\n    This is a brute force implementation that goes over all elements of the\n    group and checks for membership in the centralizer. It is used to\n    test ``.centralizer()`` from ``sympy.combinatorics.perm_groups``.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.testutil import _naive_list_centralizer\n    >>> from sympy.combinatorics.named_groups import DihedralGroup\n    >>> D = DihedralGroup(4)\n    >>> _naive_list_centralizer(D, D)\n    [Permutation([0, 1, 2, 3]), Permutation([2, 3, 0, 1])]\n\n    See Also\n    ========\n\n    sympy.combinatorics.perm_groups.centralizer\n\n    '
    from sympy.combinatorics.permutations import _af_commutes_with
    if hasattr(other, 'generators'):
        elements = list(self.generate_dimino(af=True))
        gens = [x._array_form for x in other.generators]
        commutes_with_gens = lambda x: all((_af_commutes_with(x, gen) for gen in gens))
        centralizer_list = []
        if not af:
            for element in elements:
                if commutes_with_gens(element):
                    centralizer_list.append(Permutation._af_new(element))
        else:
            for element in elements:
                if commutes_with_gens(element):
                    centralizer_list.append(element)
        return centralizer_list
    elif hasattr(other, 'getitem'):
        return _naive_list_centralizer(self, PermutationGroup(other), af)
    elif hasattr(other, 'array_form'):
        return _naive_list_centralizer(self, PermutationGroup([other]), af)
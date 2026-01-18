from sympy.core import S
from sympy.core.sympify import _sympify
from sympy.functions import KroneckerDelta
from .matexpr import MatrixExpr
from .special import ZeroMatrix, Identity, OneMatrix
def _eval_rewrite_as_BlockDiagMatrix(self, *args, **kwargs):
    from sympy.combinatorics.permutations import Permutation
    from .blockmatrix import BlockDiagMatrix
    perm = self.args[0]
    full_cyclic_form = perm.full_cyclic_form
    cycles_picks = []
    a, b, c = (0, 0, 0)
    flag = False
    for cycle in full_cyclic_form:
        l = len(cycle)
        m = max(cycle)
        if not flag:
            if m + 1 > a + l:
                flag = True
                temp = [cycle]
                b = m
                c = l
            else:
                cycles_picks.append([cycle])
                a += l
        elif m > b:
            if m + 1 == a + c + l:
                temp.append(cycle)
                cycles_picks.append(temp)
                flag = False
                a = m + 1
            else:
                b = m
                temp.append(cycle)
                c += l
        elif b + 1 == a + c + l:
            temp.append(cycle)
            cycles_picks.append(temp)
            flag = False
            a = b + 1
        else:
            temp.append(cycle)
            c += l
    p = 0
    args = []
    for pick in cycles_picks:
        new_cycles = []
        l = 0
        for cycle in pick:
            new_cycle = [i - p for i in cycle]
            new_cycles.append(new_cycle)
            l += len(cycle)
        p += l
        perm = Permutation(new_cycles)
        mat = PermutationMatrix(perm)
        args.append(mat)
    return BlockDiagMatrix(*args)
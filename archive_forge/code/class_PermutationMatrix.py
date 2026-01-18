from sympy.core import S
from sympy.core.sympify import _sympify
from sympy.functions import KroneckerDelta
from .matexpr import MatrixExpr
from .special import ZeroMatrix, Identity, OneMatrix
class PermutationMatrix(MatrixExpr):
    """A Permutation Matrix

    Parameters
    ==========

    perm : Permutation
        The permutation the matrix uses.

        The size of the permutation determines the matrix size.

        See the documentation of
        :class:`sympy.combinatorics.permutations.Permutation` for
        the further information of how to create a permutation object.

    Examples
    ========

    >>> from sympy import Matrix, PermutationMatrix
    >>> from sympy.combinatorics import Permutation

    Creating a permutation matrix:

    >>> p = Permutation(1, 2, 0)
    >>> P = PermutationMatrix(p)
    >>> P = P.as_explicit()
    >>> P
    Matrix([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]])

    Permuting a matrix row and column:

    >>> M = Matrix([0, 1, 2])
    >>> Matrix(P*M)
    Matrix([
    [1],
    [2],
    [0]])

    >>> Matrix(M.T*P)
    Matrix([[2, 0, 1]])

    See Also
    ========

    sympy.combinatorics.permutations.Permutation
    """

    def __new__(cls, perm):
        from sympy.combinatorics.permutations import Permutation
        perm = _sympify(perm)
        if not isinstance(perm, Permutation):
            raise ValueError('{} must be a SymPy Permutation instance.'.format(perm))
        return super().__new__(cls, perm)

    @property
    def shape(self):
        size = self.args[0].size
        return (size, size)

    @property
    def is_Identity(self):
        return self.args[0].is_Identity

    def doit(self, **hints):
        if self.is_Identity:
            return Identity(self.rows)
        return self

    def _entry(self, i, j, **kwargs):
        perm = self.args[0]
        return KroneckerDelta(perm.apply(i), j)

    def _eval_power(self, exp):
        return PermutationMatrix(self.args[0] ** exp).doit()

    def _eval_inverse(self):
        return PermutationMatrix(self.args[0] ** (-1))
    _eval_transpose = _eval_adjoint = _eval_inverse

    def _eval_determinant(self):
        sign = self.args[0].signature()
        if sign == 1:
            return S.One
        elif sign == -1:
            return S.NegativeOne
        raise NotImplementedError

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
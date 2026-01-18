from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
from sympy.core import Basic, sympify, S
from sympy.core.mul import mul, Mul
from sympy.core.numbers import Number, Integer
from sympy.core.symbol import Dummy
from sympy.functions import adjoint
from sympy.strategies import (rm_id, unpack, typed, flatten, exhaust,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.matrices import MatrixBase
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.matrices.expressions._shape import validate_matmul_integer as validate
from .inverse import Inverse
from .matexpr import MatrixExpr
from .matpow import MatPow
from .transpose import transpose
from .permutation import PermutationMatrix
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix
class MatMul(MatrixExpr, Mul):
    """
    A product of matrix expressions

    Examples
    ========

    >>> from sympy import MatMul, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 4)
    >>> B = MatrixSymbol('B', 4, 3)
    >>> C = MatrixSymbol('C', 3, 6)
    >>> MatMul(A, B, C)
    A*B*C
    """
    is_MatMul = True
    identity = GenericIdentity()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if not args:
            return cls.identity
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))
        obj = Basic.__new__(cls, *args)
        factor, matrices = obj.as_coeff_matrices()
        if check is not None:
            sympy_deprecation_warning('Passing check to MatMul is deprecated and the check argument will be removed in a future version.', deprecated_since_version='1.11', active_deprecations_target='remove-check-argument-from-matrix-operations')
        if check is not False:
            validate(*matrices)
        if not matrices:
            return factor
        if evaluate:
            return cls._evaluate(obj)
        return obj

    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)

    @property
    def shape(self):
        matrices = [arg for arg in self.args if arg.is_Matrix]
        return (matrices[0].rows, matrices[-1].cols)

    def _entry(self, i, j, expand=True, **kwargs):
        from sympy.concrete.summations import Sum
        from sympy.matrices.immutable import ImmutableMatrix
        coeff, matrices = self.as_coeff_matrices()
        if len(matrices) == 1:
            return coeff * matrices[0][i, j]
        indices = [None] * (len(matrices) + 1)
        ind_ranges = [None] * (len(matrices) - 1)
        indices[0] = i
        indices[-1] = j

        def f():
            counter = 1
            while True:
                yield Dummy('i_%i' % counter)
                counter += 1
        dummy_generator = kwargs.get('dummy_generator', f())
        for i in range(1, len(matrices)):
            indices[i] = next(dummy_generator)
        for i, arg in enumerate(matrices[:-1]):
            ind_ranges[i] = arg.shape[1] - 1
        matrices = [arg._entry(indices[i], indices[i + 1], dummy_generator=dummy_generator) for i, arg in enumerate(matrices)]
        expr_in_sum = Mul.fromiter(matrices)
        if any((v.has(ImmutableMatrix) for v in matrices)):
            expand = True
        result = coeff * Sum(expr_in_sum, *zip(indices[1:-1], [0] * len(ind_ranges), ind_ranges))
        if not any((isinstance(v, (Integer, int)) for v in ind_ranges)):
            expand = False
        return result.doit() if expand else result

    def as_coeff_matrices(self):
        scalars = [x for x in self.args if not x.is_Matrix]
        matrices = [x for x in self.args if x.is_Matrix]
        coeff = Mul(*scalars)
        if coeff.is_commutative is False:
            raise NotImplementedError('noncommutative scalars in MatMul are not supported.')
        return (coeff, matrices)

    def as_coeff_mmul(self):
        coeff, matrices = self.as_coeff_matrices()
        return (coeff, MatMul(*matrices))

    def expand(self, **kwargs):
        expanded = super(MatMul, self).expand(**kwargs)
        return self._evaluate(expanded)

    def _eval_transpose(self):
        """Transposition of matrix multiplication.

        Notes
        =====

        The following rules are applied.

        Transposition for matrix multiplied with another matrix:
        `\\left(A B\\right)^{T} = B^{T} A^{T}`

        Transposition for matrix multiplied with scalar:
        `\\left(c A\\right)^{T} = c A^{T}`

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Transpose
        """
        coeff, matrices = self.as_coeff_matrices()
        return MatMul(coeff, *[transpose(arg) for arg in matrices[::-1]]).doit()

    def _eval_adjoint(self):
        return MatMul(*[adjoint(arg) for arg in self.args[::-1]]).doit()

    def _eval_trace(self):
        factor, mmul = self.as_coeff_mmul()
        if factor != 1:
            from .trace import trace
            return factor * trace(mmul.doit())
        else:
            raise NotImplementedError("Can't simplify any further")

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import Determinant
        factor, matrices = self.as_coeff_matrices()
        square_matrices = only_squares(*matrices)
        return factor ** self.rows * Mul(*list(map(Determinant, square_matrices)))

    def _eval_inverse(self):
        if all((arg.is_square for arg in self.args if isinstance(arg, MatrixExpr))):
            return MatMul(*(arg.inverse() if isinstance(arg, MatrixExpr) else arg ** (-1) for arg in self.args[::-1])).doit()
        return Inverse(self)

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = tuple((arg.doit(**hints) for arg in self.args))
        else:
            args = self.args
        expr = canonicalize(MatMul(*args))
        return expr

    def args_cnc(self, cset=False, warn=True, **kwargs):
        coeff_c = [x for x in self.args if x.is_commutative]
        coeff_nc = [x for x in self.args if not x.is_commutative]
        if cset:
            clen = len(coeff_c)
            coeff_c = set(coeff_c)
            if clen and warn and (len(coeff_c) != clen):
                raise ValueError('repeated commutative arguments: %s' % [ci for ci in coeff_c if list(self.args).count(ci) > 1])
        return [coeff_c, coeff_nc]

    def _eval_derivative_matrix_lines(self, x):
        from .transpose import Transpose
        with_x_ind = [i for i, arg in enumerate(self.args) if arg.has(x)]
        lines = []
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind + 1:]
            if right_args:
                right_mat = MatMul.fromiter(right_args)
            else:
                right_mat = Identity(self.shape[1])
            if left_args:
                left_rev = MatMul.fromiter([Transpose(i).doit() if i.is_Matrix else i for i in reversed(left_args)])
            else:
                left_rev = Identity(self.shape[0])
            d = self.args[ind]._eval_derivative_matrix_lines(x)
            for i in d:
                i.append_first(left_rev)
                i.append_second(right_mat)
                lines.append(i)
        return lines
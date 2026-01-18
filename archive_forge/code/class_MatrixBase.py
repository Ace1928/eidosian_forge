import mpmath as mp
from collections.abc import Callable
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import diff
from sympy.core.expr import Expr
from sympy.core.kind import _NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, uniquely_named_symbol
from sympy.core.sympify import sympify, _sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.polys import cancel
from sympy.printing import sstr
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import flatten, NotIterable, is_sequence, reshape
from sympy.utilities.misc import as_int, filldedent
from .common import (
from .utilities import _iszero, _is_zero_after_expand_mul, _simplify
from .determinant import (
from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize
from .eigen import (
from .decompositions import (
from .graph import (
from .solvers import (
from .inverse import (
class MatrixBase(MatrixDeprecated, MatrixCalculus, MatrixEigen, MatrixCommon, Printable):
    """Base class for matrix objects."""
    __array_priority__ = 11
    is_Matrix = True
    _class_priority = 3
    _sympify = staticmethod(sympify)
    zero = S.Zero
    one = S.One

    @property
    def kind(self) -> MatrixKind:
        elem_kinds = {e.kind for e in self.flat()}
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            elemkind = UndefinedKind
        return MatrixKind(elemkind)

    def flat(self):
        return [self[i, j] for i in range(self.rows) for j in range(self.cols)]

    def __array__(self, dtype=object):
        from .dense import matrix2numpy
        return matrix2numpy(self, dtype=dtype)

    def __len__(self):
        """Return the number of elements of ``self``.

        Implemented mainly so bool(Matrix()) == False.
        """
        return self.rows * self.cols

    def _matrix_pow_by_jordan_blocks(self, num):
        from sympy.matrices import diag, MutableMatrix

        def jordan_cell_power(jc, n):
            N = jc.shape[0]
            l = jc[0, 0]
            if l.is_zero:
                if N == 1 and n.is_nonnegative:
                    jc[0, 0] = l ** n
                elif not (n.is_integer and n.is_nonnegative):
                    raise NonInvertibleMatrixError('Non-invertible matrix can only be raised to a nonnegative integer')
                else:
                    for i in range(N):
                        jc[0, i] = KroneckerDelta(i, n)
            else:
                for i in range(N):
                    bn = binomial(n, i)
                    if isinstance(bn, binomial):
                        bn = bn._eval_expand_func()
                    jc[0, i] = l ** (n - i) * bn
            for i in range(N):
                for j in range(1, N - i):
                    jc[j, i + j] = jc[j - 1, i + j - 1]
        P, J = self.jordan_form()
        jordan_cells = J.get_diag_blocks()
        jordan_cells = [MutableMatrix(j) for j in jordan_cells]
        for j in jordan_cells:
            jordan_cell_power(j, num)
        return self._new(P.multiply(diag(*jordan_cells)).multiply(P.inv()))

    def __str__(self):
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        return 'Matrix(%s)' % str(self.tolist())

    def _format_str(self, printer=None):
        if not printer:
            printer = StrPrinter()
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        if self.rows == 1:
            return 'Matrix([%s])' % self.table(printer, rowsep=',\n')
        return 'Matrix([\n%s])' % self.table(printer, rowsep=',\n')

    @classmethod
    def irregular(cls, ntop, *matrices, **kwargs):
        """Return a matrix filled by the given matrices which
      are listed in order of appearance from left to right, top to
      bottom as they first appear in the matrix. They must fill the
      matrix completely.

      Examples
      ========

      >>> from sympy import ones, Matrix
      >>> Matrix.irregular(3, ones(2,1), ones(3,3)*2, ones(2,2)*3,
      ...   ones(1,1)*4, ones(2,2)*5, ones(1,2)*6, ones(1,2)*7)
      Matrix([
        [1, 2, 2, 2, 3, 3],
        [1, 2, 2, 2, 3, 3],
        [4, 2, 2, 2, 5, 5],
        [6, 6, 7, 7, 5, 5]])
      """
        ntop = as_int(ntop)
        b = [i.as_explicit() if hasattr(i, 'as_explicit') else i for i in matrices]
        q = list(range(len(b)))
        dat = [i.rows for i in b]
        active = [q.pop(0) for _ in range(ntop)]
        cols = sum([b[i].cols for i in active])
        rows = []
        while any(dat):
            r = []
            for a, j in enumerate(active):
                r.extend(b[j][-dat[j], :])
                dat[j] -= 1
                if dat[j] == 0 and q:
                    active[a] = q.pop(0)
            if len(r) != cols:
                raise ValueError(filldedent('\n                Matrices provided do not appear to fill\n                the space completely.'))
            rows.append(r)
        return cls._new(rows)

    @classmethod
    def _handle_ndarray(cls, arg):
        arr = arg.__array__()
        if len(arr.shape) == 2:
            rows, cols = (arr.shape[0], arr.shape[1])
            flat_list = [cls._sympify(i) for i in arr.ravel()]
            return (rows, cols, flat_list)
        elif len(arr.shape) == 1:
            flat_list = [cls._sympify(i) for i in arr]
            return (arr.shape[0], 1, flat_list)
        else:
            raise NotImplementedError('SymPy supports just 1D and 2D matrices')

    @classmethod
    def _handle_creation_inputs(cls, *args, **kwargs):
        """Return the number of rows, cols and flat matrix elements.

        Examples
        ========

        >>> from sympy import Matrix, I

        Matrix can be constructed as follows:

        * from a nested list of iterables

        >>> Matrix( ((1, 2+I), (3, 4)) )
        Matrix([
        [1, 2 + I],
        [3,     4]])

        * from un-nested iterable (interpreted as a column)

        >>> Matrix( [1, 2] )
        Matrix([
        [1],
        [2]])

        * from un-nested iterable with dimensions

        >>> Matrix(1, 2, [1, 2] )
        Matrix([[1, 2]])

        * from no arguments (a 0 x 0 matrix)

        >>> Matrix()
        Matrix(0, 0, [])

        * from a rule

        >>> Matrix(2, 2, lambda i, j: i/(j + 1) )
        Matrix([
        [0,   0],
        [1, 1/2]])

        See Also
        ========
        irregular - filling a matrix with irregular blocks
        """
        from sympy.matrices import SparseMatrix
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.matrices.expressions.blockmatrix import BlockMatrix
        flat_list = None
        if len(args) == 1:
            if isinstance(args[0], SparseMatrix):
                return (args[0].rows, args[0].cols, flatten(args[0].tolist()))
            elif isinstance(args[0], MatrixBase):
                return (args[0].rows, args[0].cols, args[0].flat())
            elif isinstance(args[0], Basic) and args[0].is_Matrix:
                return (args[0].rows, args[0].cols, args[0].as_explicit().flat())
            elif isinstance(args[0], mp.matrix):
                M = args[0]
                flat_list = [cls._sympify(x) for x in M]
                return (M.rows, M.cols, flat_list)
            elif hasattr(args[0], '__array__'):
                return cls._handle_ndarray(args[0])
            elif is_sequence(args[0]) and (not isinstance(args[0], DeferredVector)):
                dat = list(args[0])
                ismat = lambda i: isinstance(i, MatrixBase) and (evaluate or isinstance(i, BlockMatrix) or isinstance(i, MatrixSymbol))
                raw = lambda i: is_sequence(i) and (not ismat(i))
                evaluate = kwargs.get('evaluate', True)
                if evaluate:

                    def make_explicit(x):
                        """make Block and Symbol explicit"""
                        if isinstance(x, BlockMatrix):
                            return x.as_explicit()
                        elif isinstance(x, MatrixSymbol) and all((_.is_Integer for _ in x.shape)):
                            return x.as_explicit()
                        else:
                            return x

                    def make_explicit_row(row):
                        if isinstance(row, (list, tuple)):
                            return [make_explicit(x) for x in row]
                        else:
                            return make_explicit(row)
                    if isinstance(dat, (list, tuple)):
                        dat = [make_explicit_row(row) for row in dat]
                if dat in ([], [[]]):
                    rows = cols = 0
                    flat_list = []
                elif not any((raw(i) or ismat(i) for i in dat)):
                    flat_list = [cls._sympify(i) for i in dat]
                    rows = len(flat_list)
                    cols = 1 if rows else 0
                elif evaluate and all((ismat(i) for i in dat)):
                    ncol = {i.cols for i in dat if any(i.shape)}
                    if ncol:
                        if len(ncol) != 1:
                            raise ValueError('mismatched dimensions')
                        flat_list = [_ for i in dat for r in i.tolist() for _ in r]
                        cols = ncol.pop()
                        rows = len(flat_list) // cols
                    else:
                        rows = cols = 0
                        flat_list = []
                elif evaluate and any((ismat(i) for i in dat)):
                    ncol = set()
                    flat_list = []
                    for i in dat:
                        if ismat(i):
                            flat_list.extend([k for j in i.tolist() for k in j])
                            if any(i.shape):
                                ncol.add(i.cols)
                        elif raw(i):
                            if i:
                                ncol.add(len(i))
                                flat_list.extend([cls._sympify(ij) for ij in i])
                        else:
                            ncol.add(1)
                            flat_list.append(i)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                    cols = ncol.pop()
                    rows = len(flat_list) // cols
                else:
                    flat_list = []
                    ncol = set()
                    rows = cols = 0
                    for row in dat:
                        if not is_sequence(row) and (not getattr(row, 'is_Matrix', False)):
                            raise ValueError('expecting list of lists')
                        if hasattr(row, '__array__'):
                            if 0 in row.shape:
                                continue
                        elif not row:
                            continue
                        if evaluate and all((ismat(i) for i in row)):
                            r, c, flatT = cls._handle_creation_inputs([i.T for i in row])
                            T = reshape(flatT, [c])
                            flat = [T[i][j] for j in range(c) for i in range(r)]
                            r, c = (c, r)
                        else:
                            r = 1
                            if getattr(row, 'is_Matrix', False):
                                c = 1
                                flat = [row]
                            else:
                                c = len(row)
                                flat = [cls._sympify(i) for i in row]
                        ncol.add(c)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                        flat_list.extend(flat)
                        rows += r
                    cols = ncol.pop() if ncol else 0
        elif len(args) == 3:
            rows = as_int(args[0])
            cols = as_int(args[1])
            if rows < 0 or cols < 0:
                raise ValueError('Cannot create a {} x {} matrix. Both dimensions must be positive'.format(rows, cols))
            if len(args) == 3 and isinstance(args[2], Callable):
                op = args[2]
                flat_list = []
                for i in range(rows):
                    flat_list.extend([cls._sympify(op(cls._sympify(i), cls._sympify(j))) for j in range(cols)])
            elif len(args) == 3 and is_sequence(args[2]):
                flat_list = args[2]
                if len(flat_list) != rows * cols:
                    raise ValueError('List length should be equal to rows*columns')
                flat_list = [cls._sympify(i) for i in flat_list]
        elif len(args) == 0:
            rows = cols = 0
            flat_list = []
        if flat_list is None:
            raise TypeError(filldedent('\n                Data type not understood; expecting list of lists\n                or lists of values.'))
        return (rows, cols, flat_list)

    def _setitem(self, key, value):
        """Helper to set value at location given by key.

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        To replace row r you assign to position r*m where m
        is the number of columns:

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        And to replace column c you can assign to position c:

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        from .dense import Matrix
        is_slice = isinstance(key, slice)
        i, j = key = self.key2ij(key)
        is_mat = isinstance(value, MatrixBase)
        if isinstance(i, slice) or isinstance(j, slice):
            if is_mat:
                self.copyin_matrix(key, value)
                return
            if not isinstance(value, Expr) and is_sequence(value):
                self.copyin_list(key, value)
                return
            raise ValueError('unexpected value: %s' % value)
        else:
            if not is_mat and (not isinstance(value, Basic)) and is_sequence(value):
                value = Matrix(value)
                is_mat = True
            if is_mat:
                if is_slice:
                    key = (slice(*divmod(i, self.cols)), slice(*divmod(j, self.cols)))
                else:
                    key = (slice(i, i + value.rows), slice(j, j + value.cols))
                self.copyin_matrix(key, value)
            else:
                return (i, j, self._sympify(value))
            return

    def add(self, b):
        """Return self + b."""
        return self + b

    def condition_number(self):
        """Returns the condition number of a matrix.

        This is the maximum singular value divided by the minimum singular value

        Examples
        ========

        >>> from sympy import Matrix, S
        >>> A = Matrix([[1, 0, 0], [0, 10, 0], [0, 0, S.One/10]])
        >>> A.condition_number()
        100

        See Also
        ========

        singular_values
        """
        if not self:
            return self.zero
        singularvalues = self.singular_values()
        return Max(*singularvalues) / Min(*singularvalues)

    def copy(self):
        """
        Returns the copy of a matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.copy()
        Matrix([
        [1, 2],
        [3, 4]])

        """
        return self._new(self.rows, self.cols, self.flat())

    def cross(self, b):
        """
        Return the cross product of ``self`` and ``b`` relaxing the condition
        of compatible dimensions: if each has 3 elements, a matrix of the
        same type and shape as ``self`` will be returned. If ``b`` has the same
        shape as ``self`` then common identities for the cross product (like
        `a \\times b = - b \\times a`) will hold.

        Parameters
        ==========
            b : 3x1 or 1x3 Matrix

        See Also
        ========

        dot
        multiply
        multiply_elementwise
        """
        from sympy.matrices.expressions.matexpr import MatrixExpr
        if not isinstance(b, (MatrixBase, MatrixExpr)):
            raise TypeError('{} must be a Matrix, not {}.'.format(b, type(b)))
        if not self.rows * self.cols == b.rows * b.cols == 3:
            raise ShapeError('Dimensions incorrect for cross product: %s x %s' % ((self.rows, self.cols), (b.rows, b.cols)))
        else:
            return self._new(self.rows, self.cols, (self[1] * b[2] - self[2] * b[1], self[2] * b[0] - self[0] * b[2], self[0] * b[1] - self[1] * b[0]))

    @property
    def D(self):
        """Return Dirac conjugate (if ``self.rows == 4``).

        Examples
        ========

        >>> from sympy import Matrix, I, eye
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m.D
        Matrix([[0, 1 - I, -2, -3]])
        >>> m = (eye(4) + I*eye(4))
        >>> m[0, 3] = 2
        >>> m.D
        Matrix([
        [1 - I,     0,      0,      0],
        [    0, 1 - I,      0,      0],
        [    0,     0, -1 + I,      0],
        [    2,     0,      0, -1 + I]])

        If the matrix does not have 4 rows an AttributeError will be raised
        because this property is only defined for matrices with 4 rows.

        >>> Matrix(eye(2)).D
        Traceback (most recent call last):
        ...
        AttributeError: Matrix has no attribute D.

        See Also
        ========

        sympy.matrices.common.MatrixCommon.conjugate: By-element conjugation
        sympy.matrices.common.MatrixCommon.H: Hermite conjugation
        """
        from sympy.physics.matrices import mgamma
        if self.rows != 4:
            raise AttributeError
        return self.H * mgamma(0)

    def dot(self, b, hermitian=None, conjugate_convention=None):
        """Return the dot or inner product of two vectors of equal length.
        Here ``self`` must be a ``Matrix`` of size 1 x n or n x 1, and ``b``
        must be either a matrix of size 1 x n, n x 1, or a list/tuple of length n.
        A scalar is returned.

        By default, ``dot`` does not conjugate ``self`` or ``b``, even if there are
        complex entries. Set ``hermitian=True`` (and optionally a ``conjugate_convention``)
        to compute the hermitian inner product.

        Possible kwargs are ``hermitian`` and ``conjugate_convention``.

        If ``conjugate_convention`` is ``"left"``, ``"math"`` or ``"maths"``,
        the conjugate of the first vector (``self``) is used.  If ``"right"``
        or ``"physics"`` is specified, the conjugate of the second vector ``b`` is used.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> v = Matrix([1, 1, 1])
        >>> M.row(0).dot(v)
        6
        >>> M.col(0).dot(v)
        12
        >>> v = [3, 2, 1]
        >>> M.row(0).dot(v)
        10

        >>> from sympy import I
        >>> q = Matrix([1*I, 1*I, 1*I])
        >>> q.dot(q, hermitian=False)
        -3

        >>> q.dot(q, hermitian=True)
        3

        >>> q1 = Matrix([1, 1, 1*I])
        >>> q.dot(q1, hermitian=True, conjugate_convention="maths")
        1 - 2*I
        >>> q.dot(q1, hermitian=True, conjugate_convention="physics")
        1 + 2*I


        See Also
        ========

        cross
        multiply
        multiply_elementwise
        """
        from .dense import Matrix
        if not isinstance(b, MatrixBase):
            if is_sequence(b):
                if len(b) != self.cols and len(b) != self.rows:
                    raise ShapeError('Dimensions incorrect for dot product: %s, %s' % (self.shape, len(b)))
                return self.dot(Matrix(b))
            else:
                raise TypeError('`b` must be an ordered iterable or Matrix, not %s.' % type(b))
        if 1 not in self.shape or 1 not in b.shape:
            raise ShapeError
        if len(self) != len(b):
            raise ShapeError('Dimensions incorrect for dot product: %s, %s' % (self.shape, b.shape))
        mat = self
        n = len(mat)
        if mat.shape != (1, n):
            mat = mat.reshape(1, n)
        if b.shape != (n, 1):
            b = b.reshape(n, 1)
        if conjugate_convention is not None and hermitian is None:
            hermitian = True
        if hermitian and conjugate_convention is None:
            conjugate_convention = 'maths'
        if hermitian == True:
            if conjugate_convention in ('maths', 'left', 'math'):
                mat = mat.conjugate()
            elif conjugate_convention in ('physics', 'right'):
                b = b.conjugate()
            else:
                raise ValueError('Unknown conjugate_convention was entered. conjugate_convention must be one of the following: math, maths, left, physics or right.')
        return (mat * b)[0]

    def dual(self):
        """Returns the dual of a matrix.

        A dual of a matrix is:

        ``(1/2)*levicivita(i, j, k, l)*M(k, l)`` summed over indices `k` and `l`

        Since the levicivita method is anti_symmetric for any pairwise
        exchange of indices, the dual of a symmetric matrix is the zero
        matrix. Strictly speaking the dual defined here assumes that the
        'matrix' `M` is a contravariant anti_symmetric second rank tensor,
        so that the dual is a covariant second rank tensor.

        """
        from sympy.matrices import zeros
        M, n = (self[:, :], self.rows)
        work = zeros(n)
        if self.is_symmetric():
            return work
        for i in range(1, n):
            for j in range(1, n):
                acum = 0
                for k in range(1, n):
                    acum += LeviCivita(i, j, 0, k) * M[0, k]
                work[i, j] = acum
                work[j, i] = -acum
        for l in range(1, n):
            acum = 0
            for a in range(1, n):
                for b in range(1, n):
                    acum += LeviCivita(0, l, a, b) * M[a, b]
            acum /= 2
            work[0, l] = -acum
            work[l, 0] = acum
        return work

    def _eval_matrix_exp_jblock(self):
        """A helper function to compute an exponential of a Jordan block
        matrix

        Examples
        ========

        >>> from sympy import Symbol, Matrix
        >>> l = Symbol('lamda')

        A trivial example of 1*1 Jordan block:

        >>> m = Matrix.jordan_block(1, l)
        >>> m._eval_matrix_exp_jblock()
        Matrix([[exp(lamda)]])

        An example of 3*3 Jordan block:

        >>> m = Matrix.jordan_block(3, l)
        >>> m._eval_matrix_exp_jblock()
        Matrix([
        [exp(lamda), exp(lamda), exp(lamda)/2],
        [         0, exp(lamda),   exp(lamda)],
        [         0,          0,   exp(lamda)]])

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Matrix_function#Jordan_decomposition
        """
        size = self.rows
        l = self[0, 0]
        exp_l = exp(l)
        bands = {i: exp_l / factorial(i) for i in range(size)}
        from .sparsetools import banded
        return self.__class__(banded(size, bands))

    def analytic_func(self, f, x):
        """
        Computes f(A) where A is a Square Matrix
        and f is an analytic function.

        Examples
        ========

        >>> from sympy import Symbol, Matrix, S, log

        >>> x = Symbol('x')
        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])
        >>> f = log(x)
        >>> m.analytic_func(f, x)
        Matrix([
        [     0, log(2)],
        [log(2),      0]])

        Parameters
        ==========

        f : Expr
            Analytic Function
        x : Symbol
            parameter of f

        """
        f, x = (_sympify(f), _sympify(x))
        if not self.is_square:
            raise NonSquareMatrixError
        if not x.is_symbol:
            raise ValueError('{} must be a symbol.'.format(x))
        if x not in f.free_symbols:
            raise ValueError('{} must be a parameter of {}.'.format(x, f))
        if x in self.free_symbols:
            raise ValueError('{} must not be a parameter of {}.'.format(x, self))
        eigen = self.eigenvals()
        max_mul = max(eigen.values())
        derivative = {}
        dd = f
        for i in range(max_mul - 1):
            dd = diff(dd, x)
            derivative[i + 1] = dd
        n = self.shape[0]
        r = self.zeros(n)
        f_val = self.zeros(n, 1)
        row = 0
        for i in eigen:
            mul = eigen[i]
            f_val[row] = f.subs(x, i)
            if f_val[row].is_number and (not f_val[row].is_complex):
                raise ValueError('Cannot evaluate the function because the function {} is not analytic at the given eigenvalue {}'.format(f, f_val[row]))
            val = 1
            for a in range(n):
                r[row, a] = val
                val *= i
            if mul > 1:
                coe = [1 for ii in range(n)]
                deri = 1
                while mul > 1:
                    row = row + 1
                    mul -= 1
                    d_i = derivative[deri].subs(x, i)
                    if d_i.is_number and (not d_i.is_complex):
                        raise ValueError('Cannot evaluate the function because the derivative {} is not analytic at the given eigenvalue {}'.format(derivative[deri], d_i))
                    f_val[row] = d_i
                    for a in range(n):
                        if a - deri + 1 <= 0:
                            r[row, a] = 0
                            coe[a] = 0
                            continue
                        coe[a] = coe[a] * (a - deri + 1)
                        r[row, a] = coe[a] * pow(i, a - deri)
                    deri += 1
            row += 1
        c = r.solve(f_val)
        ans = self.zeros(n)
        pre = self.eye(n)
        for i in range(n):
            ans = ans + c[i] * pre
            pre *= self
        return ans

    def exp(self):
        """Return the exponential of a square matrix.

        Examples
        ========

        >>> from sympy import Symbol, Matrix

        >>> t = Symbol('t')
        >>> m = Matrix([[0, 1], [-1, 0]]) * t
        >>> m.exp()
        Matrix([
        [    exp(I*t)/2 + exp(-I*t)/2, -I*exp(I*t)/2 + I*exp(-I*t)/2],
        [I*exp(I*t)/2 - I*exp(-I*t)/2,      exp(I*t)/2 + exp(-I*t)/2]])
        """
        if not self.is_square:
            raise NonSquareMatrixError('Exponentiation is valid only for square matrices')
        try:
            P, J = self.jordan_form()
            cells = J.get_diag_blocks()
        except MatrixError:
            raise NotImplementedError('Exponentiation is implemented only for matrices for which the Jordan normal form can be computed')
        blocks = [cell._eval_matrix_exp_jblock() for cell in cells]
        from sympy.matrices import diag
        eJ = diag(*blocks)
        ret = P.multiply(eJ, dotprodsimp=None).multiply(P.inv(), dotprodsimp=None)
        if all((value.is_real for value in self.values())):
            return type(self)(re(ret))
        else:
            return type(self)(ret)

    def _eval_matrix_log_jblock(self):
        """Helper function to compute logarithm of a jordan block.

        Examples
        ========

        >>> from sympy import Symbol, Matrix
        >>> l = Symbol('lamda')

        A trivial example of 1*1 Jordan block:

        >>> m = Matrix.jordan_block(1, l)
        >>> m._eval_matrix_log_jblock()
        Matrix([[log(lamda)]])

        An example of 3*3 Jordan block:

        >>> m = Matrix.jordan_block(3, l)
        >>> m._eval_matrix_log_jblock()
        Matrix([
        [log(lamda),    1/lamda, -1/(2*lamda**2)],
        [         0, log(lamda),         1/lamda],
        [         0,          0,      log(lamda)]])
        """
        size = self.rows
        l = self[0, 0]
        if l.is_zero:
            raise MatrixError('Could not take logarithm or reciprocal for the given eigenvalue {}'.format(l))
        bands = {0: log(l)}
        for i in range(1, size):
            bands[i] = -(-l) ** (-i) / i
        from .sparsetools import banded
        return self.__class__(banded(size, bands))

    def log(self, simplify=cancel):
        """Return the logarithm of a square matrix.

        Parameters
        ==========

        simplify : function, bool
            The function to simplify the result with.

            Default is ``cancel``, which is effective to reduce the
            expression growing for taking reciprocals and inverses for
            symbolic matrices.

        Examples
        ========

        >>> from sympy import S, Matrix

        Examples for positive-definite matrices:

        >>> m = Matrix([[1, 1], [0, 1]])
        >>> m.log()
        Matrix([
        [0, 1],
        [0, 0]])

        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])
        >>> m.log()
        Matrix([
        [     0, log(2)],
        [log(2),      0]])

        Examples for non positive-definite matrices:

        >>> m = Matrix([[S(3)/4, S(5)/4], [S(5)/4, S(3)/4]])
        >>> m.log()
        Matrix([
        [         I*pi/2, log(2) - I*pi/2],
        [log(2) - I*pi/2,          I*pi/2]])

        >>> m = Matrix(
        ...     [[0, 0, 0, 1],
        ...      [0, 0, 1, 0],
        ...      [0, 1, 0, 0],
        ...      [1, 0, 0, 0]])
        >>> m.log()
        Matrix([
        [ I*pi/2,       0,       0, -I*pi/2],
        [      0,  I*pi/2, -I*pi/2,       0],
        [      0, -I*pi/2,  I*pi/2,       0],
        [-I*pi/2,       0,       0,  I*pi/2]])
        """
        if not self.is_square:
            raise NonSquareMatrixError('Logarithm is valid only for square matrices')
        try:
            if simplify:
                P, J = simplify(self).jordan_form()
            else:
                P, J = self.jordan_form()
            cells = J.get_diag_blocks()
        except MatrixError:
            raise NotImplementedError('Logarithm is implemented only for matrices for which the Jordan normal form can be computed')
        blocks = [cell._eval_matrix_log_jblock() for cell in cells]
        from sympy.matrices import diag
        eJ = diag(*blocks)
        if simplify:
            ret = simplify(P * eJ * simplify(P.inv()))
            ret = self.__class__(ret)
        else:
            ret = P * eJ * P.inv()
        return ret

    def is_nilpotent(self):
        """Checks if a matrix is nilpotent.

        A matrix B is nilpotent if for some integer k, B**k is
        a zero matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        >>> a.is_nilpotent()
        True

        >>> a = Matrix([[1, 0, 1], [1, 0, 0], [1, 1, 0]])
        >>> a.is_nilpotent()
        False
        """
        if not self:
            return True
        if not self.is_square:
            raise NonSquareMatrixError('Nilpotency is valid only for square matrices')
        x = uniquely_named_symbol('x', self, modify=lambda s: '_' + s)
        p = self.charpoly(x)
        if p.args[0] == x ** self.rows:
            return True
        return False

    def key2bounds(self, keys):
        """Converts a key with potentially mixed types of keys (integer and slice)
        into a tuple of ranges and raises an error if any index is out of ``self``'s
        range.

        See Also
        ========

        key2ij
        """
        islice, jslice = [isinstance(k, slice) for k in keys]
        if islice:
            if not self.rows:
                rlo = rhi = 0
            else:
                rlo, rhi = keys[0].indices(self.rows)[:2]
        else:
            rlo = a2idx(keys[0], self.rows)
            rhi = rlo + 1
        if jslice:
            if not self.cols:
                clo = chi = 0
            else:
                clo, chi = keys[1].indices(self.cols)[:2]
        else:
            clo = a2idx(keys[1], self.cols)
            chi = clo + 1
        return (rlo, rhi, clo, chi)

    def key2ij(self, key):
        """Converts key into canonical form, converting integers or indexable
        items into valid integers for ``self``'s range or returning slices
        unchanged.

        See Also
        ========

        key2bounds
        """
        if is_sequence(key):
            if not len(key) == 2:
                raise TypeError('key must be a sequence of length 2')
            return [a2idx(i, n) if not isinstance(i, slice) else i for i, n in zip(key, self.shape)]
        elif isinstance(key, slice):
            return key.indices(len(self))[:2]
        else:
            return divmod(a2idx(key, len(self)), self.cols)

    def normalized(self, iszerofunc=_iszero):
        """Return the normalized version of ``self``.

        Parameters
        ==========

        iszerofunc : Function, optional
            A function to determine whether ``self`` is a zero vector.
            The default ``_iszero`` tests to see if each element is
            exactly zero.

        Returns
        =======

        Matrix
            Normalized vector form of ``self``.
            It has the same length as a unit vector. However, a zero vector
            will be returned for a vector with norm 0.

        Raises
        ======

        ShapeError
            If the matrix is not in a vector form.

        See Also
        ========

        norm
        """
        if self.rows != 1 and self.cols != 1:
            raise ShapeError('A Matrix must be a vector to normalize.')
        norm = self.norm()
        if iszerofunc(norm):
            out = self.zeros(self.rows, self.cols)
        else:
            out = self.applyfunc(lambda i: i / norm)
        return out

    def norm(self, ord=None):
        """Return the Norm of a Matrix or Vector.

        In the simplest case this is the geometric size of the vector
        Other norms can be specified by the ord parameter


        =====  ============================  ==========================
        ord    norm for matrices             norm for vectors
        =====  ============================  ==========================
        None   Frobenius norm                2-norm
        'fro'  Frobenius norm                - does not exist
        inf    maximum row sum               max(abs(x))
        -inf   --                            min(abs(x))
        1      maximum column sum            as below
        -1     --                            as below
        2      2-norm (largest sing. value)  as below
        -2     smallest singular value       as below
        other  - does not exist              sum(abs(x)**ord)**(1./ord)
        =====  ============================  ==========================

        Examples
        ========

        >>> from sympy import Matrix, Symbol, trigsimp, cos, sin, oo
        >>> x = Symbol('x', real=True)
        >>> v = Matrix([cos(x), sin(x)])
        >>> trigsimp( v.norm() )
        1
        >>> v.norm(10)
        (sin(x)**10 + cos(x)**10)**(1/10)
        >>> A = Matrix([[1, 1], [1, 1]])
        >>> A.norm(1) # maximum sum of absolute values of A is 2
        2
        >>> A.norm(2) # Spectral norm (max of |Ax|/|x| under 2-vector-norm)
        2
        >>> A.norm(-2) # Inverse spectral norm (smallest singular value)
        0
        >>> A.norm() # Frobenius Norm
        2
        >>> A.norm(oo) # Infinity Norm
        2
        >>> Matrix([1, -2]).norm(oo)
        2
        >>> Matrix([-1, 2]).norm(-oo)
        1

        See Also
        ========

        normalized
        """
        vals = list(self.values()) or [0]
        if S.One in self.shape:
            if ord in (2, None):
                return sqrt(Add(*(abs(i) ** 2 for i in vals)))
            elif ord == 1:
                return Add(*(abs(i) for i in vals))
            elif ord is S.Infinity:
                return Max(*[abs(i) for i in vals])
            elif ord is S.NegativeInfinity:
                return Min(*[abs(i) for i in vals])
            try:
                return Pow(Add(*(abs(i) ** ord for i in vals)), S.One / ord)
            except (NotImplementedError, TypeError):
                raise ValueError('Expected order to be Number, Symbol, oo')
        elif ord == 1:
            m = self.applyfunc(abs)
            return Max(*[sum(m.col(i)) for i in range(m.cols)])
        elif ord == 2:
            return Max(*self.singular_values())
        elif ord == -2:
            return Min(*self.singular_values())
        elif ord is S.Infinity:
            m = self.applyfunc(abs)
            return Max(*[sum(m.row(i)) for i in range(m.rows)])
        elif ord is None or (isinstance(ord, str) and ord.lower() in ['f', 'fro', 'frobenius', 'vector']):
            return self.vec().norm(ord=2)
        else:
            raise NotImplementedError('Matrix Norms under development')

    def print_nonzero(self, symb='X'):
        """Shows location of non-zero entries for fast shape lookup.

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> m = Matrix(2, 3, lambda i, j: i*3+j)
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5]])
        >>> m.print_nonzero()
        [ XX]
        [XXX]
        >>> m = eye(4)
        >>> m.print_nonzero("x")
        [x   ]
        [ x  ]
        [  x ]
        [   x]

        """
        s = []
        for i in range(self.rows):
            line = []
            for j in range(self.cols):
                if self[i, j] == 0:
                    line.append(' ')
                else:
                    line.append(str(symb))
            s.append('[%s]' % ''.join(line))
        print('\n'.join(s))

    def project(self, v):
        """Return the projection of ``self`` onto the line containing ``v``.

        Examples
        ========

        >>> from sympy import Matrix, S, sqrt
        >>> V = Matrix([sqrt(3)/2, S.Half])
        >>> x = Matrix([[1, 0]])
        >>> V.project(x)
        Matrix([[sqrt(3)/2, 0]])
        >>> V.project(-x)
        Matrix([[sqrt(3)/2, 0]])
        """
        return v * (self.dot(v) / v.dot(v))

    def table(self, printer, rowstart='[', rowend=']', rowsep='\n', colsep=', ', align='right'):
        """
        String form of Matrix as a table.

        ``printer`` is the printer to use for on the elements (generally
        something like StrPrinter())

        ``rowstart`` is the string used to start each row (by default '[').

        ``rowend`` is the string used to end each row (by default ']').

        ``rowsep`` is the string used to separate rows (by default a newline).

        ``colsep`` is the string used to separate columns (by default ', ').

        ``align`` defines how the elements are aligned. Must be one of 'left',
        'right', or 'center'.  You can also use '<', '>', and '^' to mean the
        same thing, respectively.

        This is used by the string printer for Matrix.

        Examples
        ========

        >>> from sympy import Matrix, StrPrinter
        >>> M = Matrix([[1, 2], [-33, 4]])
        >>> printer = StrPrinter()
        >>> M.table(printer)
        '[  1, 2]\\n[-33, 4]'
        >>> print(M.table(printer))
        [  1, 2]
        [-33, 4]
        >>> print(M.table(printer, rowsep=',\\n'))
        [  1, 2],
        [-33, 4]
        >>> print('[%s]' % M.table(printer, rowsep=',\\n'))
        [[  1, 2],
        [-33, 4]]
        >>> print(M.table(printer, colsep=' '))
        [  1 2]
        [-33 4]
        >>> print(M.table(printer, align='center'))
        [ 1 , 2]
        [-33, 4]
        >>> print(M.table(printer, rowstart='{', rowend='}'))
        {  1, 2}
        {-33, 4}
        """
        if S.Zero in self.shape:
            return '[]'
        res = []
        maxlen = [0] * self.cols
        for i in range(self.rows):
            res.append([])
            for j in range(self.cols):
                s = printer._print(self[i, j])
                res[-1].append(s)
                maxlen[j] = max(len(s), maxlen[j])
        align = {'left': 'ljust', 'right': 'rjust', 'center': 'center', '<': 'ljust', '>': 'rjust', '^': 'center'}[align]
        for i, row in enumerate(res):
            for j, elem in enumerate(row):
                row[j] = getattr(elem, align)(maxlen[j])
            res[i] = rowstart + colsep.join(row) + rowend
        return rowsep.join(res)

    def rank_decomposition(self, iszerofunc=_iszero, simplify=False):
        return _rank_decomposition(self, iszerofunc=iszerofunc, simplify=simplify)

    def cholesky(self, hermitian=True):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LDLdecomposition(self, hermitian=True):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LUdecomposition(self, iszerofunc=_iszero, simpfunc=None, rankcheck=False):
        return _LUdecomposition(self, iszerofunc=iszerofunc, simpfunc=simpfunc, rankcheck=rankcheck)

    def LUdecomposition_Simple(self, iszerofunc=_iszero, simpfunc=None, rankcheck=False):
        return _LUdecomposition_Simple(self, iszerofunc=iszerofunc, simpfunc=simpfunc, rankcheck=rankcheck)

    def LUdecompositionFF(self):
        return _LUdecompositionFF(self)

    def singular_value_decomposition(self):
        return _singular_value_decomposition(self)

    def QRdecomposition(self):
        return _QRdecomposition(self)

    def upper_hessenberg_decomposition(self):
        return _upper_hessenberg_decomposition(self)

    def diagonal_solve(self, rhs):
        return _diagonal_solve(self, rhs)

    def lower_triangular_solve(self, rhs):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def upper_triangular_solve(self, rhs):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def cholesky_solve(self, rhs):
        return _cholesky_solve(self, rhs)

    def LDLsolve(self, rhs):
        return _LDLsolve(self, rhs)

    def LUsolve(self, rhs, iszerofunc=_iszero):
        return _LUsolve(self, rhs, iszerofunc=iszerofunc)

    def QRsolve(self, b):
        return _QRsolve(self, b)

    def gauss_jordan_solve(self, B, freevar=False):
        return _gauss_jordan_solve(self, B, freevar=freevar)

    def pinv_solve(self, B, arbitrary_matrix=None):
        return _pinv_solve(self, B, arbitrary_matrix=arbitrary_matrix)

    def solve(self, rhs, method='GJ'):
        return _solve(self, rhs, method=method)

    def solve_least_squares(self, rhs, method='CH'):
        return _solve_least_squares(self, rhs, method=method)

    def pinv(self, method='RD'):
        return _pinv(self, method=method)

    def inv_mod(self, m):
        return _inv_mod(self, m)

    def inverse_ADJ(self, iszerofunc=_iszero):
        return _inv_ADJ(self, iszerofunc=iszerofunc)

    def inverse_BLOCK(self, iszerofunc=_iszero):
        return _inv_block(self, iszerofunc=iszerofunc)

    def inverse_GE(self, iszerofunc=_iszero):
        return _inv_GE(self, iszerofunc=iszerofunc)

    def inverse_LU(self, iszerofunc=_iszero):
        return _inv_LU(self, iszerofunc=iszerofunc)

    def inverse_CH(self, iszerofunc=_iszero):
        return _inv_CH(self, iszerofunc=iszerofunc)

    def inverse_LDL(self, iszerofunc=_iszero):
        return _inv_LDL(self, iszerofunc=iszerofunc)

    def inverse_QR(self, iszerofunc=_iszero):
        return _inv_QR(self, iszerofunc=iszerofunc)

    def inv(self, method=None, iszerofunc=_iszero, try_block_diag=False):
        return _inv(self, method=method, iszerofunc=iszerofunc, try_block_diag=try_block_diag)

    def connected_components(self):
        return _connected_components(self)

    def connected_components_decomposition(self):
        return _connected_components_decomposition(self)

    def strongly_connected_components(self):
        return _strongly_connected_components(self)

    def strongly_connected_components_decomposition(self, lower=True):
        return _strongly_connected_components_decomposition(self, lower=lower)
    _sage_ = Basic._sage_
    rank_decomposition.__doc__ = _rank_decomposition.__doc__
    cholesky.__doc__ = _cholesky.__doc__
    LDLdecomposition.__doc__ = _LDLdecomposition.__doc__
    LUdecomposition.__doc__ = _LUdecomposition.__doc__
    LUdecomposition_Simple.__doc__ = _LUdecomposition_Simple.__doc__
    LUdecompositionFF.__doc__ = _LUdecompositionFF.__doc__
    singular_value_decomposition.__doc__ = _singular_value_decomposition.__doc__
    QRdecomposition.__doc__ = _QRdecomposition.__doc__
    upper_hessenberg_decomposition.__doc__ = _upper_hessenberg_decomposition.__doc__
    diagonal_solve.__doc__ = _diagonal_solve.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__
    cholesky_solve.__doc__ = _cholesky_solve.__doc__
    LDLsolve.__doc__ = _LDLsolve.__doc__
    LUsolve.__doc__ = _LUsolve.__doc__
    QRsolve.__doc__ = _QRsolve.__doc__
    gauss_jordan_solve.__doc__ = _gauss_jordan_solve.__doc__
    pinv_solve.__doc__ = _pinv_solve.__doc__
    solve.__doc__ = _solve.__doc__
    solve_least_squares.__doc__ = _solve_least_squares.__doc__
    pinv.__doc__ = _pinv.__doc__
    inv_mod.__doc__ = _inv_mod.__doc__
    inverse_ADJ.__doc__ = _inv_ADJ.__doc__
    inverse_GE.__doc__ = _inv_GE.__doc__
    inverse_LU.__doc__ = _inv_LU.__doc__
    inverse_CH.__doc__ = _inv_CH.__doc__
    inverse_LDL.__doc__ = _inv_LDL.__doc__
    inverse_QR.__doc__ = _inv_QR.__doc__
    inverse_BLOCK.__doc__ = _inv_block.__doc__
    inv.__doc__ = _inv.__doc__
    connected_components.__doc__ = _connected_components.__doc__
    connected_components_decomposition.__doc__ = _connected_components_decomposition.__doc__
    strongly_connected_components.__doc__ = _strongly_connected_components.__doc__
    strongly_connected_components_decomposition.__doc__ = _strongly_connected_components_decomposition.__doc__
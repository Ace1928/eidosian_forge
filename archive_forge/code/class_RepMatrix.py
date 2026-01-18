from collections import defaultdict
from operator import index as index_
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer, Rational
from sympy.core.sympify import _sympify, SympifyError
from sympy.core.singleton import S
from sympy.polys.domains import ZZ, QQ, EXRAW
from sympy.polys.matrices import DomainMatrix
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent
from .common import classof
from .matrices import MatrixBase, MatrixKind, ShapeError
class RepMatrix(MatrixBase):
    """Matrix implementation based on DomainMatrix as an internal representation.

    The RepMatrix class is a superclass for Matrix, ImmutableMatrix,
    SparseMatrix and ImmutableSparseMatrix which are the main usable matrix
    classes in SymPy. Most methods on this class are simply forwarded to
    DomainMatrix.
    """
    _rep: DomainMatrix

    def __eq__(self, other):
        if not isinstance(other, RepMatrix):
            try:
                other = _sympify(other)
            except SympifyError:
                return NotImplemented
            if not isinstance(other, RepMatrix):
                return NotImplemented
        return self._rep.unify_eq(other._rep)

    @classmethod
    def _unify_element_sympy(cls, rep, element):
        domain = rep.domain
        element = _sympify(element)
        if domain != EXRAW:
            if element.is_Integer:
                new_domain = domain
            elif element.is_Rational:
                new_domain = QQ
            else:
                new_domain = EXRAW
            if new_domain != domain:
                rep = rep.convert_to(new_domain)
                domain = new_domain
            if domain != EXRAW:
                element = new_domain.from_sympy(element)
        if domain == EXRAW and (not isinstance(element, Expr)):
            sympy_deprecation_warning('\n                non-Expr objects in a Matrix is deprecated. Matrix represents\n                a mathematical matrix. To represent a container of non-numeric\n                entities, Use a list of lists, TableForm, NumPy array, or some\n                other data structure instead.\n                ', deprecated_since_version='1.9', active_deprecations_target='deprecated-non-expr-in-matrix', stacklevel=4)
        return (rep, element)

    @classmethod
    def _dod_to_DomainMatrix(cls, rows, cols, dod, types):
        if not all((issubclass(typ, Expr) for typ in types)):
            sympy_deprecation_warning('\n                non-Expr objects in a Matrix is deprecated. Matrix represents\n                a mathematical matrix. To represent a container of non-numeric\n                entities, Use a list of lists, TableForm, NumPy array, or some\n                other data structure instead.\n                ', deprecated_since_version='1.9', active_deprecations_target='deprecated-non-expr-in-matrix', stacklevel=6)
        rep = DomainMatrix(dod, (rows, cols), EXRAW)
        if all((issubclass(typ, Rational) for typ in types)):
            if all((issubclass(typ, Integer) for typ in types)):
                rep = rep.convert_to(ZZ)
            else:
                rep = rep.convert_to(QQ)
        return rep

    @classmethod
    def _flat_list_to_DomainMatrix(cls, rows, cols, flat_list):
        elements_dod = defaultdict(dict)
        for n, element in enumerate(flat_list):
            if element != 0:
                i, j = divmod(n, cols)
                elements_dod[i][j] = element
        types = set(map(type, flat_list))
        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    @classmethod
    def _smat_to_DomainMatrix(cls, rows, cols, smat):
        elements_dod = defaultdict(dict)
        for (i, j), element in smat.items():
            if element != 0:
                elements_dod[i][j] = element
        types = set(map(type, smat.values()))
        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    def flat(self):
        return self._rep.to_sympy().to_list_flat()

    def _eval_tolist(self):
        return self._rep.to_sympy().to_list()

    def _eval_todok(self):
        return self._rep.to_sympy().to_dok()

    def _eval_values(self):
        return list(self.todok().values())

    def copy(self):
        return self._fromrep(self._rep.copy())

    @property
    def kind(self) -> MatrixKind:
        domain = self._rep.domain
        element_kind: Kind
        if domain in (ZZ, QQ):
            element_kind = NumberKind
        elif domain == EXRAW:
            kinds = {e.kind for e in self.values()}
            if len(kinds) == 1:
                [element_kind] = kinds
            else:
                element_kind = UndefinedKind
        else:
            raise RuntimeError('Domain should only be ZZ, QQ or EXRAW')
        return MatrixKind(element_kind)

    def _eval_has(self, *patterns):
        zhas = False
        dok = self.todok()
        if len(dok) != self.rows * self.cols:
            zhas = S.Zero.has(*patterns)
        return zhas or any((value.has(*patterns) for value in dok.values()))

    def _eval_is_Identity(self):
        if not all((self[i, i] == 1 for i in range(self.rows))):
            return False
        return len(self.todok()) == self.rows

    def _eval_is_symmetric(self, simpfunc):
        diff = (self - self.T).applyfunc(simpfunc)
        return len(diff.values()) == 0

    def _eval_transpose(self):
        """Returns the transposed SparseMatrix of this SparseMatrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a = SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.T
        Matrix([
        [1, 3],
        [2, 4]])
        """
        return self._fromrep(self._rep.transpose())

    def _eval_col_join(self, other):
        return self._fromrep(self._rep.vstack(other._rep))

    def _eval_row_join(self, other):
        return self._fromrep(self._rep.hstack(other._rep))

    def _eval_extract(self, rowsList, colsList):
        return self._fromrep(self._rep.extract(rowsList, colsList))

    def __getitem__(self, key):
        return _getitem_RepMatrix(self, key)

    @classmethod
    def _eval_zeros(cls, rows, cols):
        rep = DomainMatrix.zeros((rows, cols), ZZ)
        return cls._fromrep(rep)

    @classmethod
    def _eval_eye(cls, rows, cols):
        rep = DomainMatrix.eye((rows, cols), ZZ)
        return cls._fromrep(rep)

    def _eval_add(self, other):
        return classof(self, other)._fromrep(self._rep + other._rep)

    def _eval_matrix_mul(self, other):
        return classof(self, other)._fromrep(self._rep * other._rep)

    def _eval_matrix_mul_elementwise(self, other):
        selfrep, otherrep = self._rep.unify(other._rep)
        newrep = selfrep.mul_elementwise(otherrep)
        return classof(self, other)._fromrep(newrep)

    def _eval_scalar_mul(self, other):
        rep, other = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.scalarmul(other))

    def _eval_scalar_rmul(self, other):
        rep, other = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.rscalarmul(other))

    def _eval_Abs(self):
        return self._fromrep(self._rep.applyfunc(abs))

    def _eval_conjugate(self):
        rep = self._rep
        domain = rep.domain
        if domain in (ZZ, QQ):
            return self.copy()
        else:
            return self._fromrep(rep.applyfunc(lambda e: e.conjugate()))

    def equals(self, other, failing_expression=False):
        """Applies ``equals`` to corresponding elements of the matrices,
        trying to prove that the elements are equivalent, returning True
        if they are, False if any pair is not, and None (or the first
        failing expression if failing_expression is True) if it cannot
        be decided if the expressions are equivalent or not. This is, in
        general, an expensive operation.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x
        >>> A = Matrix([x*(x - 1), 0])
        >>> B = Matrix([x**2 - x, 0])
        >>> A == B
        False
        >>> A.simplify() == B.simplify()
        True
        >>> A.equals(B)
        True
        >>> A.equals(2)
        False

        See Also
        ========
        sympy.core.expr.Expr.equals
        """
        if self.shape != getattr(other, 'shape', None):
            return False
        rv = True
        for i in range(self.rows):
            for j in range(self.cols):
                ans = self[i, j].equals(other[i, j], failing_expression)
                if ans is False:
                    return False
                elif ans is not True and rv is True:
                    rv = ans
        return rv
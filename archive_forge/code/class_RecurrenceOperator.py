from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify
class RecurrenceOperator:
    """
    The Recurrence Operators are defined by a list of polynomials
    in the base ring and the parent ring of the Operator.

    Explanation
    ===========

    Takes a list of polynomials for each power of Sn and the
    parent ring which must be an instance of RecurrenceOperatorAlgebra.

    A Recurrence Operator can be created easily using
    the operator `Sn`. See examples below.

    Examples
    ========

    >>> from sympy.holonomic.recurrence import RecurrenceOperator, RecurrenceOperators
    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n),'Sn')

    >>> RecurrenceOperator([0, 1, n**2], R)
    (1)Sn + (n**2)Sn**2

    >>> Sn*n
    (n + 1)Sn

    >>> n*Sn*n + 1 - Sn**2*n
    (1) + (n**2 + n)Sn + (-n - 2)Sn**2

    See Also
    ========

    DifferentialOperatorAlgebra
    """
    _op_priority = 20

    def __init__(self, list_of_poly, parent):
        self.parent = parent
        if isinstance(list_of_poly, list):
            for i, j in enumerate(list_of_poly):
                if isinstance(j, int):
                    list_of_poly[i] = self.parent.base.from_sympy(S(j))
                elif not isinstance(j, self.parent.base.dtype):
                    list_of_poly[i] = self.parent.base.from_sympy(j)
            self.listofpoly = list_of_poly
        self.order = len(self.listofpoly) - 1

    def __mul__(self, other):
        """
        Multiplies two Operators and returns another
        RecurrenceOperator instance using the commutation rule
        Sn * a(n) = a(n + 1) * Sn
        """
        listofself = self.listofpoly
        base = self.parent.base
        if not isinstance(other, RecurrenceOperator):
            if not isinstance(other, self.parent.base.dtype):
                listofother = [self.parent.base.from_sympy(sympify(other))]
            else:
                listofother = [other]
        else:
            listofother = other.listofpoly

        def _mul_dmp_diffop(b, listofother):
            if isinstance(listofother, list):
                sol = []
                for i in listofother:
                    sol.append(i * b)
                return sol
            else:
                return [b * listofother]
        sol = _mul_dmp_diffop(listofself[0], listofother)

        def _mul_Sni_b(b):
            sol = [base.zero]
            if isinstance(b, list):
                for i in b:
                    j = base.to_sympy(i).subs(base.gens[0], base.gens[0] + S.One)
                    sol.append(base.from_sympy(j))
            else:
                j = b.subs(base.gens[0], base.gens[0] + S.One)
                sol.append(base.from_sympy(j))
            return sol
        for i in range(1, len(listofself)):
            listofother = _mul_Sni_b(listofother)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))
        return RecurrenceOperator(sol, self.parent)

    def __rmul__(self, other):
        if not isinstance(other, RecurrenceOperator):
            if isinstance(other, int):
                other = S(other)
            if not isinstance(other, self.parent.base.dtype):
                other = self.parent.base.from_sympy(other)
            sol = []
            for j in self.listofpoly:
                sol.append(other * j)
            return RecurrenceOperator(sol, self.parent)

    def __add__(self, other):
        if isinstance(other, RecurrenceOperator):
            sol = _add_lists(self.listofpoly, other.listofpoly)
            return RecurrenceOperator(sol, self.parent)
        else:
            if isinstance(other, int):
                other = S(other)
            list_self = self.listofpoly
            if not isinstance(other, self.parent.base.dtype):
                list_other = [self.parent.base.from_sympy(other)]
            else:
                list_other = [other]
            sol = []
            sol.append(list_self[0] + list_other[0])
            sol += list_self[1:]
            return RecurrenceOperator(sol, self.parent)
    __radd__ = __add__

    def __sub__(self, other):
        return self + -1 * other

    def __rsub__(self, other):
        return -1 * self + other

    def __pow__(self, n):
        if n == 1:
            return self
        if n == 0:
            return RecurrenceOperator([self.parent.base.one], self.parent)
        if self.listofpoly == self.parent.shift_operator.listofpoly:
            sol = []
            for i in range(0, n):
                sol.append(self.parent.base.zero)
            sol.append(self.parent.base.one)
            return RecurrenceOperator(sol, self.parent)
        elif n % 2 == 1:
            powreduce = self ** (n - 1)
            return powreduce * self
        elif n % 2 == 0:
            powreduce = self ** (n / 2)
            return powreduce * powreduce

    def __str__(self):
        listofpoly = self.listofpoly
        print_str = ''
        for i, j in enumerate(listofpoly):
            if j == self.parent.base.zero:
                continue
            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue
            if print_str:
                print_str += ' + '
            if i == 1:
                print_str += '(' + sstr(j) + ')Sn'
                continue
            print_str += '(' + sstr(j) + ')' + 'Sn**' + sstr(i)
        return print_str
    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, RecurrenceOperator):
            if self.listofpoly == other.listofpoly and self.parent == other.parent:
                return True
            else:
                return False
        elif self.listofpoly[0] == other:
            for i in self.listofpoly[1:]:
                if i is not self.parent.base.zero:
                    return False
            return True
        else:
            return False
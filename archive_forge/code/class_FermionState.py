from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
class FermionState(FockState):
    """
    Base class for FockStateFermion(Ket/Bra).
    """
    fermi_level = 0

    def __new__(cls, occupations, fermi_level=0):
        occupations = list(map(sympify, occupations))
        if len(occupations) > 1:
            try:
                occupations, sign = _sort_anticommuting_fermions(occupations, key=hash)
            except ViolationOfPauliPrinciple:
                return S.Zero
        else:
            sign = 0
        cls.fermi_level = fermi_level
        if cls._count_holes(occupations) > fermi_level:
            return S.Zero
        if sign % 2:
            return S.NegativeOne * FockState.__new__(cls, occupations)
        else:
            return FockState.__new__(cls, occupations)

    def up(self, i):
        """
        Performs the action of a creation operator.

        Explanation
        ===========

        If below fermi we try to remove a hole,
        if above fermi we try to create a particle.

        If general index p we return ``Kronecker(p,i)*self``
        where ``i`` is a new symbol with restriction above or below.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import FKet
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> FKet([]).up(a)
        FockStateFermionKet((a,))

        A creator acting on vacuum below fermi vanishes

        >>> FKet([]).up(i)
        0


        """
        present = i in self.args[0]
        if self._only_above_fermi(i):
            if present:
                return S.Zero
            else:
                return self._add_orbit(i)
        elif self._only_below_fermi(i):
            if present:
                return self._remove_orbit(i)
            else:
                return S.Zero
        elif present:
            hole = Dummy('i', below_fermi=True)
            return KroneckerDelta(i, hole) * self._remove_orbit(i)
        else:
            particle = Dummy('a', above_fermi=True)
            return KroneckerDelta(i, particle) * self._add_orbit(i)

    def down(self, i):
        """
        Performs the action of an annihilation operator.

        Explanation
        ===========

        If below fermi we try to create a hole,
        If above fermi we try to remove a particle.

        If general index p we return ``Kronecker(p,i)*self``
        where ``i`` is a new symbol with restriction above or below.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import FKet
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        An annihilator acting on vacuum above fermi vanishes

        >>> FKet([]).down(a)
        0

        Also below fermi, it vanishes, unless we specify a fermi level > 0

        >>> FKet([]).down(i)
        0
        >>> FKet([],4).down(i)
        FockStateFermionKet((i,))

        """
        present = i in self.args[0]
        if self._only_above_fermi(i):
            if present:
                return self._remove_orbit(i)
            else:
                return S.Zero
        elif self._only_below_fermi(i):
            if present:
                return S.Zero
            else:
                return self._add_orbit(i)
        elif present:
            hole = Dummy('i', below_fermi=True)
            return KroneckerDelta(i, hole) * self._add_orbit(i)
        else:
            particle = Dummy('a', above_fermi=True)
            return KroneckerDelta(i, particle) * self._remove_orbit(i)

    @classmethod
    def _only_below_fermi(cls, i):
        """
        Tests if given orbit is only below fermi surface.

        If nothing can be concluded we return a conservative False.
        """
        if i.is_number:
            return i <= cls.fermi_level
        if i.assumptions0.get('below_fermi'):
            return True
        return False

    @classmethod
    def _only_above_fermi(cls, i):
        """
        Tests if given orbit is only above fermi surface.

        If fermi level has not been set we return True.
        If nothing can be concluded we return a conservative False.
        """
        if i.is_number:
            return i > cls.fermi_level
        if i.assumptions0.get('above_fermi'):
            return True
        return not cls.fermi_level

    def _remove_orbit(self, i):
        """
        Removes particle/fills hole in orbit i. No input tests performed here.
        """
        new_occs = list(self.args[0])
        pos = new_occs.index(i)
        del new_occs[pos]
        if pos % 2:
            return S.NegativeOne * self.__class__(new_occs, self.fermi_level)
        else:
            return self.__class__(new_occs, self.fermi_level)

    def _add_orbit(self, i):
        """
        Adds particle/creates hole in orbit i. No input tests performed here.
        """
        return self.__class__((i,) + self.args[0], self.fermi_level)

    @classmethod
    def _count_holes(cls, list):
        """
        Returns the number of identified hole states in list.
        """
        return len([i for i in list if cls._only_below_fermi(i)])

    def _negate_holes(self, list):
        return tuple([-i if i <= self.fermi_level else i for i in list])

    def __repr__(self):
        if self.fermi_level:
            return 'FockStateKet(%r, fermi_level=%s)' % (self.args[0], self.fermi_level)
        else:
            return 'FockStateKet(%r)' % (self.args[0],)

    def _labels(self):
        return self._negate_holes(self.args[0])
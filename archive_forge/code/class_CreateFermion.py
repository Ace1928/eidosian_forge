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
class CreateFermion(FermionicOperator, Creator):
    """
    Fermionic creation operator.
    """
    op_symbol = 'f+'

    def _dagger_(self):
        return AnnihilateFermion(self.state)

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, Dagger, BKet
        >>> from sympy.abc import x, y, n
        >>> Dagger(B(x)).apply_operator(y)
        y*CreateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))
        """
        if isinstance(state, FockStateFermionKet):
            element = self.state
            return state.up(element)
        elif isinstance(state, Mul):
            c_part, nc_part = state.args_cnc()
            if isinstance(nc_part[0], FockStateFermionKet):
                element = self.state
                return Mul(*c_part + [nc_part[0].up(element)] + nc_part[1:])
        return Mul(self, state)

    @property
    def is_q_creator(self):
        """
        Can we create a quasi-particle?  (create hole or create particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_q_creator
        1
        >>> Fd(i).is_q_creator
        0
        >>> Fd(p).is_q_creator
        1

        """
        if self.is_above_fermi:
            return 1
        return 0

    @property
    def is_q_annihilator(self):
        """
        Can we destroy a quasi-particle?  (annihilate hole or annihilate particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=1)
        >>> i = Symbol('i', below_fermi=1)
        >>> p = Symbol('p')

        >>> Fd(a).is_q_annihilator
        0
        >>> Fd(i).is_q_annihilator
        -1
        >>> Fd(p).is_q_annihilator
        -1

        """
        if self.is_below_fermi:
            return -1
        return 0

    @property
    def is_only_q_creator(self):
        """
        Always create a quasi-particle?  (create hole or create particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_only_q_creator
        True
        >>> Fd(i).is_only_q_creator
        False
        >>> Fd(p).is_only_q_creator
        False

        """
        return self.is_only_above_fermi

    @property
    def is_only_q_annihilator(self):
        """
        Always destroy a quasi-particle?  (annihilate hole or annihilate particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_only_q_annihilator
        False
        >>> Fd(i).is_only_q_annihilator
        True
        >>> Fd(p).is_only_q_annihilator
        False

        """
        return self.is_only_below_fermi

    def __repr__(self):
        return 'CreateFermion(%s)' % self.state

    def _latex(self, printer):
        if self.state is S.Zero:
            return '{a^\\dagger_{0}}'
        else:
            return '{a^\\dagger_{%s}}' % self.state.name
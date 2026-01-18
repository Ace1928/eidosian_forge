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
class NO(Expr):
    """
    This Object is used to represent normal ordering brackets.

    i.e.  {abcd}  sometimes written  :abcd:

    Explanation
    ===========

    Applying the function NO(arg) to an argument means that all operators in
    the argument will be assumed to anticommute, and have vanishing
    contractions.  This allows an immediate reordering to canonical form
    upon object creation.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import NO, F, Fd
    >>> p,q = symbols('p,q')
    >>> NO(Fd(p)*F(q))
    NO(CreateFermion(p)*AnnihilateFermion(q))
    >>> NO(F(q)*Fd(p))
    -NO(CreateFermion(p)*AnnihilateFermion(q))


    Note
    ====

    If you want to generate a normal ordered equivalent of an expression, you
    should use the function wicks().  This class only indicates that all
    operators inside the brackets anticommute, and have vanishing contractions.
    Nothing more, nothing less.

    """
    is_commutative = False

    def __new__(cls, arg):
        """
        Use anticommutation to get canonical form of operators.

        Explanation
        ===========

        Employ associativity of normal ordered product: {ab{cd}} = {abcd}
        but note that {ab}{cd} /= {abcd}.

        We also employ distributivity: {ab + cd} = {ab} + {cd}.

        Canonical form also implies expand() {ab(c+d)} = {abc} + {abd}.

        """
        arg = sympify(arg)
        arg = arg.expand()
        if arg.is_Add:
            return Add(*[cls(term) for term in arg.args])
        if arg.is_Mul:
            c_part, seq = arg.args_cnc()
            if c_part:
                coeff = Mul(*c_part)
                if not seq:
                    return coeff
            else:
                coeff = S.One
            newseq = []
            foundit = False
            for fac in seq:
                if isinstance(fac, NO):
                    newseq.extend(fac.args)
                    foundit = True
                else:
                    newseq.append(fac)
            if foundit:
                return coeff * cls(Mul(*newseq))
            if isinstance(seq[0], BosonicOperator):
                raise NotImplementedError
            try:
                newseq, sign = _sort_anticommuting_fermions(seq)
            except ViolationOfPauliPrinciple:
                return S.Zero
            if sign % 2:
                return S.NegativeOne * coeff * cls(Mul(*newseq))
            elif sign:
                return coeff * cls(Mul(*newseq))
            else:
                pass
            if coeff != S.One:
                return coeff * cls(Mul(*newseq))
            return Expr.__new__(cls, Mul(*newseq))
        if isinstance(arg, NO):
            return arg
        return arg

    @property
    def has_q_creators(self):
        """
        Return 0 if the leftmost argument of the first argument is a not a
        q_creator, else 1 if it is above fermi or -1 if it is below fermi.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import NO, F, Fd

        >>> a = symbols('a', above_fermi=True)
        >>> i = symbols('i', below_fermi=True)
        >>> NO(Fd(a)*Fd(i)).has_q_creators
        1
        >>> NO(F(i)*F(a)).has_q_creators
        -1
        >>> NO(Fd(i)*F(a)).has_q_creators           #doctest: +SKIP
        0

        """
        return self.args[0].args[0].is_q_creator

    @property
    def has_q_annihilators(self):
        """
        Return 0 if the rightmost argument of the first argument is a not a
        q_annihilator, else 1 if it is above fermi or -1 if it is below fermi.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import NO, F, Fd

        >>> a = symbols('a', above_fermi=True)
        >>> i = symbols('i', below_fermi=True)
        >>> NO(Fd(a)*Fd(i)).has_q_annihilators
        -1
        >>> NO(F(i)*F(a)).has_q_annihilators
        1
        >>> NO(Fd(a)*F(i)).has_q_annihilators
        0

        """
        return self.args[0].args[-1].is_q_annihilator

    def doit(self, **hints):
        """
        Either removes the brackets or enables complex computations
        in its arguments.

        Examples
        ========

        >>> from sympy.physics.secondquant import NO, Fd, F
        >>> from textwrap import fill
        >>> from sympy import symbols, Dummy
        >>> p,q = symbols('p,q', cls=Dummy)
        >>> print(fill(str(NO(Fd(p)*F(q)).doit())))
        KroneckerDelta(_a, _p)*KroneckerDelta(_a,
        _q)*CreateFermion(_a)*AnnihilateFermion(_a) + KroneckerDelta(_a,
        _p)*KroneckerDelta(_i, _q)*CreateFermion(_a)*AnnihilateFermion(_i) -
        KroneckerDelta(_a, _q)*KroneckerDelta(_i,
        _p)*AnnihilateFermion(_a)*CreateFermion(_i) - KroneckerDelta(_i,
        _p)*KroneckerDelta(_i, _q)*AnnihilateFermion(_i)*CreateFermion(_i)
        """
        if hints.get('remove_brackets', True):
            return self._remove_brackets()
        else:
            return self.__new__(type(self), self.args[0].doit(**hints))

    def _remove_brackets(self):
        """
        Returns the sorted string without normal order brackets.

        The returned string have the property that no nonzero
        contractions exist.
        """
        subslist = []
        for i in self.iter_q_creators():
            if self[i].is_q_annihilator:
                assume = self[i].state.assumptions0
                if isinstance(self[i].state, Dummy):
                    assume.pop('above_fermi', None)
                    assume['below_fermi'] = True
                    below = Dummy('i', **assume)
                    assume.pop('below_fermi', None)
                    assume['above_fermi'] = True
                    above = Dummy('a', **assume)
                    cls = type(self[i])
                    split = self[i].__new__(cls, below) * KroneckerDelta(below, self[i].state) + self[i].__new__(cls, above) * KroneckerDelta(above, self[i].state)
                    subslist.append((self[i], split))
                else:
                    raise SubstitutionOfAmbigousOperatorFailed(self[i])
        if subslist:
            result = NO(self.subs(subslist))
            if isinstance(result, Add):
                return Add(*[term.doit() for term in result.args])
        else:
            return self.args[0]

    def _expand_operators(self):
        """
        Returns a sum of NO objects that contain no ambiguous q-operators.

        Explanation
        ===========

        If an index q has range both above and below fermi, the operator F(q)
        is ambiguous in the sense that it can be both a q-creator and a q-annihilator.
        If q is dummy, it is assumed to be a summation variable and this method
        rewrites it into a sum of NO terms with unambiguous operators:

        {Fd(p)*F(q)} = {Fd(a)*F(b)} + {Fd(a)*F(i)} + {Fd(j)*F(b)} -{F(i)*Fd(j)}

        where a,b are above and i,j are below fermi level.
        """
        return NO(self._remove_brackets)

    def __getitem__(self, i):
        if isinstance(i, slice):
            indices = i.indices(len(self))
            return [self.args[0].args[i] for i in range(*indices)]
        else:
            return self.args[0].args[i]

    def __len__(self):
        return len(self.args[0].args)

    def iter_q_annihilators(self):
        """
        Iterates over the annihilation operators.

        Examples
        ========

        >>> from sympy import symbols
        >>> i, j = symbols('i j', below_fermi=True)
        >>> a, b = symbols('a b', above_fermi=True)
        >>> from sympy.physics.secondquant import NO, F, Fd
        >>> no = NO(Fd(a)*F(i)*F(b)*Fd(j))

        >>> no.iter_q_creators()
        <generator object... at 0x...>
        >>> list(no.iter_q_creators())
        [0, 1]
        >>> list(no.iter_q_annihilators())
        [3, 2]

        """
        ops = self.args[0].args
        iter = range(len(ops) - 1, -1, -1)
        for i in iter:
            if ops[i].is_q_annihilator:
                yield i
            else:
                break

    def iter_q_creators(self):
        """
        Iterates over the creation operators.

        Examples
        ========

        >>> from sympy import symbols
        >>> i, j = symbols('i j', below_fermi=True)
        >>> a, b = symbols('a b', above_fermi=True)
        >>> from sympy.physics.secondquant import NO, F, Fd
        >>> no = NO(Fd(a)*F(i)*F(b)*Fd(j))

        >>> no.iter_q_creators()
        <generator object... at 0x...>
        >>> list(no.iter_q_creators())
        [0, 1]
        >>> list(no.iter_q_annihilators())
        [3, 2]

        """
        ops = self.args[0].args
        iter = range(0, len(ops))
        for i in iter:
            if ops[i].is_q_creator:
                yield i
            else:
                break

    def get_subNO(self, i):
        """
        Returns a NO() without FermionicOperator at index i.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import F, NO
        >>> p, q, r = symbols('p,q,r')

        >>> NO(F(p)*F(q)*F(r)).get_subNO(1)
        NO(AnnihilateFermion(p)*AnnihilateFermion(r))

        """
        arg0 = self.args[0]
        mul = arg0._new_rawargs(*arg0.args[:i] + arg0.args[i + 1:])
        return NO(mul)

    def _latex(self, printer):
        return '\\left\\{%s\\right\\}' % printer._print(self.args[0])

    def __repr__(self):
        return 'NO(%s)' % self.args[0]

    def __str__(self):
        return ':%s:' % self.args[0]
import random
import itertools
from typing import (Sequence as tSequence, Union as tUnion, List as tList,
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, igcd, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.logic.boolalg import (And, Not, Or)
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.solvers.solveset import linsolve
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import strongly_connected_components
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import (RandomIndexedSymbol, random_symbols, RandomSymbol,
from sympy.stats.stochastic_process import StochasticPSpace
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.stats.frv_types import Bernoulli, BernoulliDistribution, FiniteRV
from sympy.stats.drv_types import Poisson, PoissonDistribution
from sympy.stats.crv_types import Normal, NormalDistribution, Gamma, GammaDistribution
from sympy.core.sympify import _sympify, sympify
class GammaProcess(CountingProcess):
    """
    A Gamma process is a random process with independent gamma distributed
    increments. It is a pure-jump increasing Levy process.

    Parameters
    ==========

    sym : Symbol/str
    lamda : Positive number
        Jump size of the process, ``lamda > 0``
    gamma : Positive number
        Rate of jump arrivals, `\\gamma > 0`

    Examples
    ========

    >>> from sympy.stats import GammaProcess, E, P, variance
    >>> from sympy import symbols, Contains, Interval, Not
    >>> t, d, x, l, g = symbols('t d x l g', positive=True)
    >>> X = GammaProcess("X", l, g)
    >>> E(X(t))
    g*t/l
    >>> variance(X(t)).simplify()
    g*t/l**2
    >>> X = GammaProcess('X', 1, 2)
    >>> P(X(t) < 1).simplify()
    lowergamma(2*t, 1)/gamma(2*t)
    >>> P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
    ... Contains(d, Interval.Lopen(7, 8))).simplify()
    -4*exp(-3) + 472*exp(-8)/3 + 1
    >>> E(X(2) + x*E(X(5)))
    10*x + 4

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_process

    """

    def __new__(cls, sym, lamda, gamma):
        _value_check(lamda > 0, 'lamda should be a positive number')
        _value_check(gamma > 0, 'gamma should be a positive number')
        sym = _symbol_converter(sym)
        gamma = _sympify(gamma)
        lamda = _sympify(lamda)
        return Basic.__new__(cls, sym, lamda, gamma)

    @property
    def lamda(self):
        return self.args[1]

    @property
    def gamma(self):
        return self.args[2]

    @property
    def state_space(self):
        return _set_converter(Interval(0, oo))

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return GammaDistribution(self.gamma * key.key, 1 / self.lamda)
        return GammaDistribution(self.gamma * key, 1 / self.lamda)

    def density(self, x):
        k = self.gamma * x.key
        theta = 1 / self.lamda
        return x ** (k - 1) * exp(-x / theta) / (gamma(k) * theta ** k)

    def simple_rv(self, rv):
        return Gamma(rv.name, self.gamma * rv.key, 1 / self.lamda)
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
class BernoulliProcess(DiscreteTimeStochasticProcess):
    """
    The Bernoulli process consists of repeated
    independent Bernoulli process trials with the same parameter `p`.
    It's assumed that the probability `p` applies to every
    trial and that the outcomes of each trial
    are independent of all the rest. Therefore Bernoulli Process
    is Discrete State and Discrete Time Stochastic Process.

    Parameters
    ==========

    sym : Symbol/str
    success : Integer/str
            The event which is considered to be success. Default: 1.
    failure: Integer/str
            The event which is considered to be failure. Default: 0.
    p : Real Number between 0 and 1
            Represents the probability of getting success.

    Examples
    ========

    >>> from sympy.stats import BernoulliProcess, P, E
    >>> from sympy import Eq, Gt
    >>> B = BernoulliProcess("B", p=0.7, success=1, failure=0)
    >>> B.state_space
    {0, 1}
    >>> (B.p).round(2)
    0.70
    >>> B.success
    1
    >>> B.failure
    0
    >>> X = B[1] + B[2] + B[3]
    >>> P(Eq(X, 0)).round(2)
    0.03
    >>> P(Eq(X, 2)).round(2)
    0.44
    >>> P(Eq(X, 4)).round(2)
    0
    >>> P(Gt(X, 1)).round(2)
    0.78
    >>> P(Eq(B[1], 0) & Eq(B[2], 1) & Eq(B[3], 0) & Eq(B[4], 1)).round(2)
    0.04
    >>> B.joint_distribution(B[1], B[2])
    JointDistributionHandmade(Lambda((B[1], B[2]), Piecewise((0.7, Eq(B[1], 1)),
    (0.3, Eq(B[1], 0)), (0, True))*Piecewise((0.7, Eq(B[2], 1)), (0.3, Eq(B[2], 0)),
    (0, True))))
    >>> E(2*B[1] + B[2]).round(2)
    2.10
    >>> P(B[1] < 1).round(2)
    0.30

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_process
    .. [2] https://mathcs.clarku.edu/~djoyce/ma217/bernoulli.pdf

    """
    index_set = S.Naturals0

    def __new__(cls, sym, p, success=1, failure=0):
        _value_check(p >= 0 and p <= 1, 'Value of p must be between 0 and 1.')
        sym = _symbol_converter(sym)
        p = _sympify(p)
        success = _sym_sympify(success)
        failure = _sym_sympify(failure)
        return Basic.__new__(cls, sym, p, success, failure)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def p(self):
        return self.args[1]

    @property
    def success(self):
        return self.args[2]

    @property
    def failure(self):
        return self.args[3]

    @property
    def state_space(self):
        return _set_converter([self.success, self.failure])

    def distribution(self, key=None):
        if key is None:
            self._deprecation_warn_distribution()
            return BernoulliDistribution(self.p)
        return BernoulliDistribution(self.p, self.success, self.failure)

    def simple_rv(self, rv):
        return Bernoulli(rv.name, p=self.p, succ=self.success, fail=self.failure)

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation.

        Parameters
        ==========

        expr : RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition : Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """
        return _SubstituteRV._expectation(expr, condition, evaluate, **kwargs)

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition : Relational
                Condition for which probability has to be computed. Must
                contain a RandomIndexedSymbol of the process.
        given_condition : Relational, Logic
                The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """
        return _SubstituteRV._probability(condition, given_condition, evaluate, **kwargs)

    def density(self, x):
        return Piecewise((self.p, Eq(x, self.success)), (1 - self.p, Eq(x, self.failure)), (S.Zero, True))
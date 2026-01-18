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
def _symbolic_probability(self, condition, new_given_condition, rv, min_key_rv):
    if isinstance(condition, Relational):
        curr_state = new_given_condition.rhs if isinstance(new_given_condition.lhs, RandomIndexedSymbol) else new_given_condition.lhs
        next_state = condition.rhs if isinstance(condition.lhs, RandomIndexedSymbol) else condition.lhs
        if isinstance(condition, (Eq, Ne)):
            if isinstance(self, DiscreteMarkovChain):
                P = self.transition_probabilities ** (rv[0].key - min_key_rv.key)
            else:
                P = exp(self.generator_matrix * (rv[0].key - min_key_rv.key))
            prob = P[curr_state, next_state] if isinstance(condition, Eq) else 1 - P[curr_state, next_state]
            return Piecewise((prob, rv[0].key > min_key_rv.key), (Probability(condition), True))
        else:
            upper = 1
            greater = False
            if isinstance(condition, (Ge, Lt)):
                upper = 0
            if isinstance(condition, (Ge, Gt)):
                greater = True
            k = Dummy('k')
            condition = Eq(condition.lhs, k) if isinstance(condition.lhs, RandomIndexedSymbol) else Eq(condition.rhs, k)
            total = Sum(self.probability(condition, new_given_condition), (k, next_state + upper, self.state_space._sup))
            return Piecewise((total, rv[0].key > min_key_rv.key), (Probability(condition), True)) if greater else Piecewise((1 - total, rv[0].key > min_key_rv.key), (Probability(condition), True))
    else:
        return Probability(condition, new_given_condition)
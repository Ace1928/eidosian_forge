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
@classmethod
def _probability(self, condition, given_condition=None, evaluate=True, **kwargs):
    """
        Internal method for computing probability of indexed RV

        Parameters
        ==========

        condition: Relational
                Condition for which probability has to be computed. Must
                contain a RandomIndexedSymbol of the process.
        given_condition: Relational/And
                The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """
    new_condition, new_givencondition = self._rvindexed_subs(condition, given_condition)
    if isinstance(new_givencondition, RandomSymbol):
        condrv = random_symbols(new_condition)
        if len(condrv) == 1 and condrv[0] == new_givencondition:
            return BernoulliDistribution(self._probability(new_condition), 0, 1)
        if any((dependent(rv, new_givencondition) for rv in condrv)):
            return Probability(new_condition, new_givencondition)
        else:
            return self._probability(new_condition)
    if new_givencondition is not None and (not isinstance(new_givencondition, (Relational, Boolean))):
        raise ValueError('%s is not a relational or combination of relationals' % new_givencondition)
    if new_givencondition == False or new_condition == False:
        return S.Zero
    if new_condition == True:
        return S.One
    if not isinstance(new_condition, (Relational, Boolean)):
        raise ValueError('%s is not a relational or combination of relationals' % new_condition)
    if new_givencondition is not None:
        return self._probability(given(new_condition, new_givencondition, **kwargs), **kwargs)
    result = pspace(new_condition).probability(new_condition, **kwargs)
    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result
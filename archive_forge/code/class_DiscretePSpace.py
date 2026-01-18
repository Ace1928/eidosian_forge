from sympy.concrete.summations import (Sum, summation)
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.polys.polytools import poly
from sympy.series.series import series
from sympy.polys.polyerrors import PolynomialError
from sympy.stats.crv import reduce_rational_inequalities_wrap
from sympy.stats.rv import (NamedArgsMixin, SinglePSpace, SingleDomain,
from sympy.stats.symbolic_probability import Probability
from sympy.sets.fancysets import Range, FiniteSet
from sympy.sets.sets import Union
from sympy.sets.contains import Contains
from sympy.utilities import filldedent
from sympy.core.sympify import _sympify
class DiscretePSpace(PSpace):
    is_real = True
    is_Discrete = True

    @property
    def pdf(self):
        return self.density(*self.symbols)

    def where(self, condition):
        rvs = random_symbols(condition)
        assert all((r.symbol in self.symbols for r in rvs))
        if len(rvs) > 1:
            raise NotImplementedError(filldedent('Multivariate discrete\n            random variables are not yet supported.'))
        conditional_domain = reduce_rational_inequalities_wrap(condition, rvs[0])
        conditional_domain = conditional_domain.intersect(self.domain.set)
        return SingleDiscreteDomain(rvs[0].symbol, conditional_domain)

    def probability(self, condition):
        complement = isinstance(condition, Ne)
        if complement:
            condition = Eq(condition.args[0], condition.args[1])
        try:
            _domain = self.where(condition).set
            if condition == False or _domain is S.EmptySet:
                return S.Zero
            if condition == True or _domain == self.domain.set:
                return S.One
            prob = self.eval_prob(_domain)
        except NotImplementedError:
            from sympy.stats.rv import density
            expr = condition.lhs - condition.rhs
            dens = density(expr)
            if not isinstance(dens, DiscreteDistribution):
                from sympy.stats.drv_types import DiscreteDistributionHandmade
                dens = DiscreteDistributionHandmade(dens)
            z = Dummy('z', real=True)
            space = SingleDiscretePSpace(z, dens)
            prob = space.probability(condition.__class__(space.value, 0))
        if prob is None:
            prob = Probability(condition)
        return prob if not complement else S.One - prob

    def eval_prob(self, _domain):
        sym = list(self.symbols)[0]
        if isinstance(_domain, Range):
            n = symbols('n', integer=True)
            inf, sup, step = (r for r in _domain.args)
            summand = self.pdf.replace(sym, n * step)
            rv = summation(summand, (n, inf / step, sup / step - 1)).doit()
            return rv
        elif isinstance(_domain, FiniteSet):
            pdf = Lambda(sym, self.pdf)
            rv = sum((pdf(x) for x in _domain))
            return rv
        elif isinstance(_domain, Union):
            rv = sum((self.eval_prob(x) for x in _domain.args))
            return rv

    def conditional_space(self, condition):
        density = Lambda(tuple(self.symbols), self.pdf / self.probability(condition))
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        domain = ConditionalDiscreteDomain(self.domain, condition)
        return DiscretePSpace(domain, density)
from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (I, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
from sympy.sets.sets import Intersection
from sympy.core.containers import Dict
from sympy.core.logic import Logic
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet
from sympy.stats.rv import (RandomDomain, ProductDomain, ConditionalDomain,
class FinitePSpace(PSpace):
    """
    A Finite Probability Space

    Represents the probabilities of a finite number of events.
    """
    is_Finite = True

    def __new__(cls, domain, density):
        density = {sympify(key): sympify(val) for key, val in density.items()}
        public_density = Dict(density)
        obj = PSpace.__new__(cls, domain, public_density)
        obj._density = density
        return obj

    def prob_of(self, elem):
        elem = sympify(elem)
        density = self._density
        if isinstance(list(density.keys())[0], FiniteSet):
            return density.get(elem, S.Zero)
        return density.get(tuple(elem)[0][1], S.Zero)

    def where(self, condition):
        assert all((r.symbol in self.symbols for r in random_symbols(condition)))
        return ConditionalFiniteDomain(self.domain, condition)

    def compute_density(self, expr):
        expr = rv_subs(expr, self.values)
        d = FiniteDensity()
        for elem in self.domain:
            val = expr.xreplace(dict(elem))
            prob = self.prob_of(elem)
            d[val] = d.get(val, S.Zero) + prob
        return d

    @cacheit
    def compute_cdf(self, expr):
        d = self.compute_density(expr)
        cum_prob = S.Zero
        cdf = []
        for key in sorted(d):
            prob = d[key]
            cum_prob += prob
            cdf.append((key, cum_prob))
        return dict(cdf)

    @cacheit
    def sorted_cdf(self, expr, python_float=False):
        cdf = self.compute_cdf(expr)
        items = list(cdf.items())
        sorted_items = sorted(items, key=lambda val_cumprob: val_cumprob[1])
        if python_float:
            sorted_items = [(v, float(cum_prob)) for v, cum_prob in sorted_items]
        return sorted_items

    @cacheit
    def compute_characteristic_function(self, expr):
        d = self.compute_density(expr)
        t = Dummy('t', real=True)
        return Lambda(t, sum((exp(I * k * t) * v for k, v in d.items())))

    @cacheit
    def compute_moment_generating_function(self, expr):
        d = self.compute_density(expr)
        t = Dummy('t', real=True)
        return Lambda(t, sum((exp(k * t) * v for k, v in d.items())))

    def compute_expectation(self, expr, rvs=None, **kwargs):
        rvs = rvs or self.values
        expr = rv_subs(expr, rvs)
        probs = [self.prob_of(elem) for elem in self.domain]
        if isinstance(expr, (Logic, Relational)):
            parse_domain = [tuple(elem)[0][1] for elem in self.domain]
            bools = [expr.xreplace(dict(elem)) for elem in self.domain]
        else:
            parse_domain = [expr.xreplace(dict(elem)) for elem in self.domain]
            bools = [True for elem in self.domain]
        return sum([Piecewise((prob * elem, blv), (S.Zero, True)) for prob, elem, blv in zip(probs, parse_domain, bools)])

    def compute_quantile(self, expr):
        cdf = self.compute_cdf(expr)
        p = Dummy('p', real=True)
        set = ((nan, (p < 0) | (p > 1)),)
        for key, value in cdf.items():
            set = set + ((key, p <= value),)
        return Lambda(p, Piecewise(*set))

    def probability(self, condition):
        cond_symbols = frozenset((rs.symbol for rs in random_symbols(condition)))
        cond = rv_subs(condition)
        if not cond_symbols.issubset(self.symbols):
            raise ValueError('Cannot compare foreign random symbols, %s' % str(cond_symbols - self.symbols))
        if isinstance(condition, Relational) and (not cond.free_symbols.issubset(self.domain.free_symbols)):
            rv = condition.lhs if isinstance(condition.rhs, Symbol) else condition.rhs
            return sum((Piecewise((self.prob_of(elem), condition.subs(rv, list(elem)[0][1])), (S.Zero, True)) for elem in self.domain))
        return sympify(sum((self.prob_of(elem) for elem in self.where(condition))))

    def conditional_space(self, condition):
        domain = self.where(condition)
        prob = self.probability(condition)
        density = {key: val / prob for key, val in self._density.items() if domain._test(key)}
        return FinitePSpace(domain, density)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library, seed)}
from __future__ import annotations
from functools import singledispatch
from math import prod
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.indexed import Indexed
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet, ProductSet, Intersection
from sympy.solvers.solveset import solveset
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
class ProductDomain(RandomDomain):
    """
    A domain resulting from the merger of two independent domains.

    See Also
    ========
    sympy.stats.crv.ProductContinuousDomain
    sympy.stats.frv.ProductFiniteDomain
    """
    is_ProductDomain = True

    def __new__(cls, *domains):
        domains2 = []
        for domain in domains:
            if not domain.is_ProductDomain:
                domains2.append(domain)
            else:
                domains2.extend(domain.domains)
        domains2 = FiniteSet(*domains2)
        if all((domain.is_Finite for domain in domains2)):
            from sympy.stats.frv import ProductFiniteDomain
            cls = ProductFiniteDomain
        if all((domain.is_Continuous for domain in domains2)):
            from sympy.stats.crv import ProductContinuousDomain
            cls = ProductContinuousDomain
        if all((domain.is_Discrete for domain in domains2)):
            from sympy.stats.drv import ProductDiscreteDomain
            cls = ProductDiscreteDomain
        return Basic.__new__(cls, *domains2)

    @property
    def sym_domain_dict(self):
        return {symbol: domain for domain in self.domains for symbol in domain.symbols}

    @property
    def symbols(self):
        return FiniteSet(*[sym for domain in self.domains for sym in domain.symbols])

    @property
    def domains(self):
        return self.args

    @property
    def set(self):
        return ProductSet(*(domain.set for domain in self.domains))

    def __contains__(self, other):
        for domain in self.domains:
            elem = frozenset([item for item in other if sympify(domain.symbols.contains(item[0])) is S.true])
            if elem not in domain:
                return False
        return True

    def as_boolean(self):
        return And(*[domain.as_boolean() for domain in self.domains])
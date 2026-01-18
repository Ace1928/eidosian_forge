from functools import wraps, reduce
from operator import mul
from typing import Optional
from sympy.core import (
from sympy.core.basic import Basic
from sympy.core.decorators import _sympifyit
from sympy.core.exprtools import Factors, factor_nc, factor_terms
from sympy.core.evalf import (
from sympy.core.function import Derivative
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.numbers import ilcm, I, Integer, equal_valued
from sympy.core.relational import Relational, Equality
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (
from sympy.polys.polyutils import (
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, public, filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, sift
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
@public
class Poly(Basic):
    """
    Generic class for representing and operating on polynomial expressions.

    See :ref:`polys-docs` for general documentation.

    Poly is a subclass of Basic rather than Expr but instances can be
    converted to Expr with the :py:meth:`~.Poly.as_expr` method.

    .. deprecated:: 1.6

       Combining Poly with non-Poly objects in binary operations is
       deprecated. Explicitly convert both objects to either Poly or Expr
       first. See :ref:`deprecated-poly-nonpoly-binary-operations`.

    Examples
    ========

    >>> from sympy import Poly
    >>> from sympy.abc import x, y

    Create a univariate polynomial:

    >>> Poly(x*(x**2 + x - 1)**2)
    Poly(x**5 + 2*x**4 - x**3 - 2*x**2 + x, x, domain='ZZ')

    Create a univariate polynomial with specific domain:

    >>> from sympy import sqrt
    >>> Poly(x**2 + 2*x + sqrt(3), domain='R')
    Poly(1.0*x**2 + 2.0*x + 1.73205080756888, x, domain='RR')

    Create a multivariate polynomial:

    >>> Poly(y*x**2 + x*y + 1)
    Poly(x**2*y + x*y + 1, x, y, domain='ZZ')

    Create a univariate polynomial, where y is a constant:

    >>> Poly(y*x**2 + x*y + 1,x)
    Poly(y*x**2 + y*x + 1, x, domain='ZZ[y]')

    You can evaluate the above polynomial as a function of y:

    >>> Poly(y*x**2 + x*y + 1,x).eval(2)
    6*y + 1

    See Also
    ========

    sympy.core.expr.Expr

    """
    __slots__ = ('rep', 'gens')
    is_commutative = True
    is_Poly = True
    _op_priority = 10.001

    def __new__(cls, rep, *gens, **args):
        """Create a new polynomial instance out of something useful. """
        opt = options.build_options(gens, args)
        if 'order' in opt:
            raise NotImplementedError("'order' keyword is not implemented yet")
        if isinstance(rep, (DMP, DMF, ANP, DomainElement)):
            return cls._from_domain_element(rep, opt)
        elif iterable(rep, exclude=str):
            if isinstance(rep, dict):
                return cls._from_dict(rep, opt)
            else:
                return cls._from_list(list(rep), opt)
        else:
            rep = sympify(rep)
            if rep.is_Poly:
                return cls._from_poly(rep, opt)
            else:
                return cls._from_expr(rep, opt)

    @classmethod
    def new(cls, rep, *gens):
        """Construct :class:`Poly` instance from raw representation. """
        if not isinstance(rep, DMP):
            raise PolynomialError('invalid polynomial representation: %s' % rep)
        elif rep.lev != len(gens) - 1:
            raise PolynomialError('invalid arguments: %s, %s' % (rep, gens))
        obj = Basic.__new__(cls)
        obj.rep = rep
        obj.gens = gens
        return obj

    @property
    def expr(self):
        return basic_from_dict(self.rep.to_sympy_dict(), *self.gens)

    @property
    def args(self):
        return (self.expr,) + self.gens

    def _hashable_content(self):
        return (self.rep,) + self.gens

    @classmethod
    def from_dict(cls, rep, *gens, **args):
        """Construct a polynomial from a ``dict``. """
        opt = options.build_options(gens, args)
        return cls._from_dict(rep, opt)

    @classmethod
    def from_list(cls, rep, *gens, **args):
        """Construct a polynomial from a ``list``. """
        opt = options.build_options(gens, args)
        return cls._from_list(rep, opt)

    @classmethod
    def from_poly(cls, rep, *gens, **args):
        """Construct a polynomial from a polynomial. """
        opt = options.build_options(gens, args)
        return cls._from_poly(rep, opt)

    @classmethod
    def from_expr(cls, rep, *gens, **args):
        """Construct a polynomial from an expression. """
        opt = options.build_options(gens, args)
        return cls._from_expr(rep, opt)

    @classmethod
    def _from_dict(cls, rep, opt):
        """Construct a polynomial from a ``dict``. """
        gens = opt.gens
        if not gens:
            raise GeneratorsNeeded("Cannot initialize from 'dict' without generators")
        level = len(gens) - 1
        domain = opt.domain
        if domain is None:
            domain, rep = construct_domain(rep, opt=opt)
        else:
            for monom, coeff in rep.items():
                rep[monom] = domain.convert(coeff)
        return cls.new(DMP.from_dict(rep, level, domain), *gens)

    @classmethod
    def _from_list(cls, rep, opt):
        """Construct a polynomial from a ``list``. """
        gens = opt.gens
        if not gens:
            raise GeneratorsNeeded("Cannot initialize from 'list' without generators")
        elif len(gens) != 1:
            raise MultivariatePolynomialError("'list' representation not supported")
        level = len(gens) - 1
        domain = opt.domain
        if domain is None:
            domain, rep = construct_domain(rep, opt=opt)
        else:
            rep = list(map(domain.convert, rep))
        return cls.new(DMP.from_list(rep, level, domain), *gens)

    @classmethod
    def _from_poly(cls, rep, opt):
        """Construct a polynomial from a polynomial. """
        if cls != rep.__class__:
            rep = cls.new(rep.rep, *rep.gens)
        gens = opt.gens
        field = opt.field
        domain = opt.domain
        if gens and rep.gens != gens:
            if set(rep.gens) != set(gens):
                return cls._from_expr(rep.as_expr(), opt)
            else:
                rep = rep.reorder(*gens)
        if 'domain' in opt and domain:
            rep = rep.set_domain(domain)
        elif field is True:
            rep = rep.to_field()
        return rep

    @classmethod
    def _from_expr(cls, rep, opt):
        """Construct a polynomial from an expression. """
        rep, opt = _dict_from_expr(rep, opt)
        return cls._from_dict(rep, opt)

    @classmethod
    def _from_domain_element(cls, rep, opt):
        gens = opt.gens
        domain = opt.domain
        level = len(gens) - 1
        rep = [domain.convert(rep)]
        return cls.new(DMP.from_list(rep, level, domain), *gens)

    def __hash__(self):
        return super().__hash__()

    @property
    def free_symbols(self):
        """
        Free symbols of a polynomial expression.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x**2 + 1).free_symbols
        {x}
        >>> Poly(x**2 + y).free_symbols
        {x, y}
        >>> Poly(x**2 + y, x).free_symbols
        {x, y}
        >>> Poly(x**2 + y, x, z).free_symbols
        {x, y}

        """
        symbols = set()
        gens = self.gens
        for i in range(len(gens)):
            for monom in self.monoms():
                if monom[i]:
                    symbols |= gens[i].free_symbols
                    break
        return symbols | self.free_symbols_in_domain

    @property
    def free_symbols_in_domain(self):
        """
        Free symbols of the domain of ``self``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 1).free_symbols_in_domain
        set()
        >>> Poly(x**2 + y).free_symbols_in_domain
        set()
        >>> Poly(x**2 + y, x).free_symbols_in_domain
        {y}

        """
        domain, symbols = (self.rep.dom, set())
        if domain.is_Composite:
            for gen in domain.symbols:
                symbols |= gen.free_symbols
        elif domain.is_EX:
            for coeff in self.coeffs():
                symbols |= coeff.free_symbols
        return symbols

    @property
    def gen(self):
        """
        Return the principal generator.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).gen
        x

        """
        return self.gens[0]

    @property
    def domain(self):
        """Get the ground domain of a :py:class:`~.Poly`

        Returns
        =======

        :py:class:`~.Domain`:
            Ground domain of the :py:class:`~.Poly`.

        Examples
        ========

        >>> from sympy import Poly, Symbol
        >>> x = Symbol('x')
        >>> p = Poly(x**2 + x)
        >>> p
        Poly(x**2 + x, x, domain='ZZ')
        >>> p.domain
        ZZ
        """
        return self.get_domain()

    @property
    def zero(self):
        """Return zero polynomial with ``self``'s properties. """
        return self.new(self.rep.zero(self.rep.lev, self.rep.dom), *self.gens)

    @property
    def one(self):
        """Return one polynomial with ``self``'s properties. """
        return self.new(self.rep.one(self.rep.lev, self.rep.dom), *self.gens)

    @property
    def unit(self):
        """Return unit polynomial with ``self``'s properties. """
        return self.new(self.rep.unit(self.rep.lev, self.rep.dom), *self.gens)

    def unify(f, g):
        """
        Make ``f`` and ``g`` belong to the same domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f, g = Poly(x/2 + 1), Poly(2*x + 1)

        >>> f
        Poly(1/2*x + 1, x, domain='QQ')
        >>> g
        Poly(2*x + 1, x, domain='ZZ')

        >>> F, G = f.unify(g)

        >>> F
        Poly(1/2*x + 1, x, domain='QQ')
        >>> G
        Poly(2*x + 1, x, domain='QQ')

        """
        _, per, F, G = f._unify(g)
        return (per(F), per(G))

    def _unify(f, g):
        g = sympify(g)
        if not g.is_Poly:
            try:
                return (f.rep.dom, f.per, f.rep, f.rep.per(f.rep.dom.from_sympy(g)))
            except CoercionFailed:
                raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        if isinstance(f.rep, DMP) and isinstance(g.rep, DMP):
            gens = _unify_gens(f.gens, g.gens)
            dom, lev = (f.rep.dom.unify(g.rep.dom, gens), len(gens) - 1)
            if f.gens != gens:
                f_monoms, f_coeffs = _dict_reorder(f.rep.to_dict(), f.gens, gens)
                if f.rep.dom != dom:
                    f_coeffs = [dom.convert(c, f.rep.dom) for c in f_coeffs]
                F = DMP(dict(list(zip(f_monoms, f_coeffs))), dom, lev)
            else:
                F = f.rep.convert(dom)
            if g.gens != gens:
                g_monoms, g_coeffs = _dict_reorder(g.rep.to_dict(), g.gens, gens)
                if g.rep.dom != dom:
                    g_coeffs = [dom.convert(c, g.rep.dom) for c in g_coeffs]
                G = DMP(dict(list(zip(g_monoms, g_coeffs))), dom, lev)
            else:
                G = g.rep.convert(dom)
        else:
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        cls = f.__class__

        def per(rep, dom=dom, gens=gens, remove=None):
            if remove is not None:
                gens = gens[:remove] + gens[remove + 1:]
                if not gens:
                    return dom.to_sympy(rep)
            return cls.new(rep, *gens)
        return (dom, per, F, G)

    def per(f, rep, gens=None, remove=None):
        """
        Create a Poly out of the given representation.

        Examples
        ========

        >>> from sympy import Poly, ZZ
        >>> from sympy.abc import x, y

        >>> from sympy.polys.polyclasses import DMP

        >>> a = Poly(x**2 + 1)

        >>> a.per(DMP([ZZ(1), ZZ(1)], ZZ), gens=[y])
        Poly(y + 1, y, domain='ZZ')

        """
        if gens is None:
            gens = f.gens
        if remove is not None:
            gens = gens[:remove] + gens[remove + 1:]
            if not gens:
                return f.rep.dom.to_sympy(rep)
        return f.__class__.new(rep, *gens)

    def set_domain(f, domain):
        """Set the ground domain of ``f``. """
        opt = options.build_options(f.gens, {'domain': domain})
        return f.per(f.rep.convert(opt.domain))

    def get_domain(f):
        """Get the ground domain of ``f``. """
        return f.rep.dom

    def set_modulus(f, modulus):
        """
        Set the modulus of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(5*x**2 + 2*x - 1, x).set_modulus(2)
        Poly(x**2 + 1, x, modulus=2)

        """
        modulus = options.Modulus.preprocess(modulus)
        return f.set_domain(FF(modulus))

    def get_modulus(f):
        """
        Get the modulus of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, modulus=2).get_modulus()
        2

        """
        domain = f.get_domain()
        if domain.is_FiniteField:
            return Integer(domain.characteristic())
        else:
            raise PolynomialError('not a polynomial over a Galois field')

    def _eval_subs(f, old, new):
        """Internal implementation of :func:`subs`. """
        if old in f.gens:
            if new.is_number:
                return f.eval(old, new)
            else:
                try:
                    return f.replace(old, new)
                except PolynomialError:
                    pass
        return f.as_expr().subs(old, new)

    def exclude(f):
        """
        Remove unnecessary generators from ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import a, b, c, d, x

        >>> Poly(a + x, a, b, c, d, x).exclude()
        Poly(a + x, a, x, domain='ZZ')

        """
        J, new = f.rep.exclude()
        gens = [gen for j, gen in enumerate(f.gens) if j not in J]
        return f.per(new, gens=gens)

    def replace(f, x, y=None, **_ignore):
        """
        Replace ``x`` with ``y`` in generators list.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 1, x).replace(x, y)
        Poly(y**2 + 1, y, domain='ZZ')

        """
        if y is None:
            if f.is_univariate:
                x, y = (f.gen, x)
            else:
                raise PolynomialError('syntax supported only in univariate case')
        if x == y or x not in f.gens:
            return f
        if x in f.gens and y not in f.gens:
            dom = f.get_domain()
            if not dom.is_Composite or y not in dom.symbols:
                gens = list(f.gens)
                gens[gens.index(x)] = y
                return f.per(f.rep, gens=gens)
        raise PolynomialError('Cannot replace %s with %s in %s' % (x, y, f))

    def match(f, *args, **kwargs):
        """Match expression from Poly. See Basic.match()"""
        return f.as_expr().match(*args, **kwargs)

    def reorder(f, *gens, **args):
        """
        Efficiently apply new order of generators.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x*y**2, x, y).reorder(y, x)
        Poly(y**2*x + x**2, y, x, domain='ZZ')

        """
        opt = options.Options((), args)
        if not gens:
            gens = _sort_gens(f.gens, opt=opt)
        elif set(f.gens) != set(gens):
            raise PolynomialError('generators list can differ only up to order of elements')
        rep = dict(list(zip(*_dict_reorder(f.rep.to_dict(), f.gens, gens))))
        return f.per(DMP(rep, f.rep.dom, len(gens) - 1), gens=gens)

    def ltrim(f, gen):
        """
        Remove dummy generators from ``f`` that are to the left of
        specified ``gen`` in the generators as ordered. When ``gen``
        is an integer, it refers to the generator located at that
        position within the tuple of generators of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(y**2 + y*z**2, x, y, z).ltrim(y)
        Poly(y**2 + y*z**2, y, z, domain='ZZ')
        >>> Poly(z, x, y, z).ltrim(-1)
        Poly(z, z, domain='ZZ')

        """
        rep = f.as_dict(native=True)
        j = f._gen_to_level(gen)
        terms = {}
        for monom, coeff in rep.items():
            if any(monom[:j]):
                raise PolynomialError('Cannot left trim %s' % f)
            terms[monom[j:]] = coeff
        gens = f.gens[j:]
        return f.new(DMP.from_dict(terms, len(gens) - 1, f.rep.dom), *gens)

    def has_only_gens(f, *gens):
        """
        Return ``True`` if ``Poly(f, *gens)`` retains ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x*y + 1, x, y, z).has_only_gens(x, y)
        True
        >>> Poly(x*y + z, x, y, z).has_only_gens(x, y)
        False

        """
        indices = set()
        for gen in gens:
            try:
                index = f.gens.index(gen)
            except ValueError:
                raise GeneratorsError("%s doesn't have %s as generator" % (f, gen))
            else:
                indices.add(index)
        for monom in f.monoms():
            for i, elt in enumerate(monom):
                if i not in indices and elt:
                    return False
        return True

    def to_ring(f):
        """
        Make the ground domain a ring.

        Examples
        ========

        >>> from sympy import Poly, QQ
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, domain=QQ).to_ring()
        Poly(x**2 + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'to_ring'):
            result = f.rep.to_ring()
        else:
            raise OperationNotSupported(f, 'to_ring')
        return f.per(result)

    def to_field(f):
        """
        Make the ground domain a field.

        Examples
        ========

        >>> from sympy import Poly, ZZ
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x, domain=ZZ).to_field()
        Poly(x**2 + 1, x, domain='QQ')

        """
        if hasattr(f.rep, 'to_field'):
            result = f.rep.to_field()
        else:
            raise OperationNotSupported(f, 'to_field')
        return f.per(result)

    def to_exact(f):
        """
        Make the ground domain exact.

        Examples
        ========

        >>> from sympy import Poly, RR
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1.0, x, domain=RR).to_exact()
        Poly(x**2 + 1, x, domain='QQ')

        """
        if hasattr(f.rep, 'to_exact'):
            result = f.rep.to_exact()
        else:
            raise OperationNotSupported(f, 'to_exact')
        return f.per(result)

    def retract(f, field=None):
        """
        Recalculate the ground domain of a polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(x**2 + 1, x, domain='QQ[y]')
        >>> f
        Poly(x**2 + 1, x, domain='QQ[y]')

        >>> f.retract()
        Poly(x**2 + 1, x, domain='ZZ')
        >>> f.retract(field=True)
        Poly(x**2 + 1, x, domain='QQ')

        """
        dom, rep = construct_domain(f.as_dict(zero=True), field=field, composite=f.domain.is_Composite or None)
        return f.from_dict(rep, f.gens, domain=dom)

    def slice(f, x, m, n=None):
        """Take a continuous subsequence of terms of ``f``. """
        if n is None:
            j, m, n = (0, x, m)
        else:
            j = f._gen_to_level(x)
        m, n = (int(m), int(n))
        if hasattr(f.rep, 'slice'):
            result = f.rep.slice(m, n, j)
        else:
            raise OperationNotSupported(f, 'slice')
        return f.per(result)

    def coeffs(f, order=None):
        """
        Returns all non-zero coefficients from ``f`` in lex order.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x + 3, x).coeffs()
        [1, 2, 3]

        See Also
        ========
        all_coeffs
        coeff_monomial
        nth

        """
        return [f.rep.dom.to_sympy(c) for c in f.rep.coeffs(order=order)]

    def monoms(f, order=None):
        """
        Returns all non-zero monomials from ``f`` in lex order.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x*y**2 + x*y + 3*y, x, y).monoms()
        [(2, 0), (1, 2), (1, 1), (0, 1)]

        See Also
        ========
        all_monoms

        """
        return f.rep.monoms(order=order)

    def terms(f, order=None):
        """
        Returns all non-zero terms from ``f`` in lex order.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x*y**2 + x*y + 3*y, x, y).terms()
        [((2, 0), 1), ((1, 2), 2), ((1, 1), 1), ((0, 1), 3)]

        See Also
        ========
        all_terms

        """
        return [(m, f.rep.dom.to_sympy(c)) for m, c in f.rep.terms(order=order)]

    def all_coeffs(f):
        """
        Returns all coefficients from a univariate polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x - 1, x).all_coeffs()
        [1, 0, 2, -1]

        """
        return [f.rep.dom.to_sympy(c) for c in f.rep.all_coeffs()]

    def all_monoms(f):
        """
        Returns all monomials from a univariate polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x - 1, x).all_monoms()
        [(3,), (2,), (1,), (0,)]

        See Also
        ========
        all_terms

        """
        return f.rep.all_monoms()

    def all_terms(f):
        """
        Returns all terms from a univariate polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x - 1, x).all_terms()
        [((3,), 1), ((2,), 0), ((1,), 2), ((0,), -1)]

        """
        return [(m, f.rep.dom.to_sympy(c)) for m, c in f.rep.all_terms()]

    def termwise(f, func, *gens, **args):
        """
        Apply a function to all terms of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> def func(k, coeff):
        ...     k = k[0]
        ...     return coeff//10**(2-k)

        >>> Poly(x**2 + 20*x + 400).termwise(func)
        Poly(x**2 + 2*x + 4, x, domain='ZZ')

        """
        terms = {}
        for monom, coeff in f.terms():
            result = func(monom, coeff)
            if isinstance(result, tuple):
                monom, coeff = result
            else:
                coeff = result
            if coeff:
                if monom not in terms:
                    terms[monom] = coeff
                else:
                    raise PolynomialError('%s monomial was generated twice' % monom)
        return f.from_dict(terms, *(gens or f.gens), **args)

    def length(f):
        """
        Returns the number of non-zero terms in ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 2*x - 1).length()
        3

        """
        return len(f.as_dict())

    def as_dict(f, native=False, zero=False):
        """
        Switch to a ``dict`` representation.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x*y**2 - y, x, y).as_dict()
        {(0, 1): -1, (1, 2): 2, (2, 0): 1}

        """
        if native:
            return f.rep.to_dict(zero=zero)
        else:
            return f.rep.to_sympy_dict(zero=zero)

    def as_list(f, native=False):
        """Switch to a ``list`` representation. """
        if native:
            return f.rep.to_list()
        else:
            return f.rep.to_sympy_list()

    def as_expr(f, *gens):
        """
        Convert a Poly instance to an Expr instance.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2 + 2*x*y**2 - y, x, y)

        >>> f.as_expr()
        x**2 + 2*x*y**2 - y
        >>> f.as_expr({x: 5})
        10*y**2 - y + 25
        >>> f.as_expr(5, 6)
        379

        """
        if not gens:
            return f.expr
        if len(gens) == 1 and isinstance(gens[0], dict):
            mapping = gens[0]
            gens = list(f.gens)
            for gen, value in mapping.items():
                try:
                    index = gens.index(gen)
                except ValueError:
                    raise GeneratorsError("%s doesn't have %s as generator" % (f, gen))
                else:
                    gens[index] = value
        return basic_from_dict(f.rep.to_sympy_dict(), *gens)

    def as_poly(self, *gens, **args):
        """Converts ``self`` to a polynomial or returns ``None``.

        >>> from sympy import sin
        >>> from sympy.abc import x, y

        >>> print((x**2 + x*y).as_poly())
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + x*y).as_poly(x, y))
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + sin(y)).as_poly(x, y))
        None

        """
        try:
            poly = Poly(self, *gens, **args)
            if not poly.is_Poly:
                return None
            else:
                return poly
        except PolynomialError:
            return None

    def lift(f):
        """
        Convert algebraic coefficients to rationals.

        Examples
        ========

        >>> from sympy import Poly, I
        >>> from sympy.abc import x

        >>> Poly(x**2 + I*x + 1, x, extension=I).lift()
        Poly(x**4 + 3*x**2 + 1, x, domain='QQ')

        """
        if hasattr(f.rep, 'lift'):
            result = f.rep.lift()
        else:
            raise OperationNotSupported(f, 'lift')
        return f.per(result)

    def deflate(f):
        """
        Reduce degree of ``f`` by mapping ``x_i**m`` to ``y_i``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**6*y**2 + x**3 + 1, x, y).deflate()
        ((3, 2), Poly(x**2*y + x + 1, x, y, domain='ZZ'))

        """
        if hasattr(f.rep, 'deflate'):
            J, result = f.rep.deflate()
        else:
            raise OperationNotSupported(f, 'deflate')
        return (J, f.per(result))

    def inject(f, front=False):
        """
        Inject ground domain generators into ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x)

        >>> f.inject()
        Poly(x**2*y + x*y**3 + x*y + 1, x, y, domain='ZZ')
        >>> f.inject(front=True)
        Poly(y**3*x + y*x**2 + y*x + 1, y, x, domain='ZZ')

        """
        dom = f.rep.dom
        if dom.is_Numerical:
            return f
        elif not dom.is_Poly:
            raise DomainError('Cannot inject generators over %s' % dom)
        if hasattr(f.rep, 'inject'):
            result = f.rep.inject(front=front)
        else:
            raise OperationNotSupported(f, 'inject')
        if front:
            gens = dom.symbols + f.gens
        else:
            gens = f.gens + dom.symbols
        return f.new(result, *gens)

    def eject(f, *gens):
        """
        Eject selected generators into the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x, y)

        >>> f.eject(x)
        Poly(x*y**3 + (x**2 + x)*y + 1, y, domain='ZZ[x]')
        >>> f.eject(y)
        Poly(y*x**2 + (y**3 + y)*x + 1, x, domain='ZZ[y]')

        """
        dom = f.rep.dom
        if not dom.is_Numerical:
            raise DomainError('Cannot eject generators over %s' % dom)
        k = len(gens)
        if f.gens[:k] == gens:
            _gens, front = (f.gens[k:], True)
        elif f.gens[-k:] == gens:
            _gens, front = (f.gens[:-k], False)
        else:
            raise NotImplementedError('can only eject front or back generators')
        dom = dom.inject(*gens)
        if hasattr(f.rep, 'eject'):
            result = f.rep.eject(dom, front=front)
        else:
            raise OperationNotSupported(f, 'eject')
        return f.new(result, *_gens)

    def terms_gcd(f):
        """
        Remove GCD of terms from the polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**6*y**2 + x**3*y, x, y).terms_gcd()
        ((3, 1), Poly(x**3*y + 1, x, y, domain='ZZ'))

        """
        if hasattr(f.rep, 'terms_gcd'):
            J, result = f.rep.terms_gcd()
        else:
            raise OperationNotSupported(f, 'terms_gcd')
        return (J, f.per(result))

    def add_ground(f, coeff):
        """
        Add an element of the ground domain to ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).add_ground(2)
        Poly(x + 3, x, domain='ZZ')

        """
        if hasattr(f.rep, 'add_ground'):
            result = f.rep.add_ground(coeff)
        else:
            raise OperationNotSupported(f, 'add_ground')
        return f.per(result)

    def sub_ground(f, coeff):
        """
        Subtract an element of the ground domain from ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).sub_ground(2)
        Poly(x - 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'sub_ground'):
            result = f.rep.sub_ground(coeff)
        else:
            raise OperationNotSupported(f, 'sub_ground')
        return f.per(result)

    def mul_ground(f, coeff):
        """
        Multiply ``f`` by a an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 1).mul_ground(2)
        Poly(2*x + 2, x, domain='ZZ')

        """
        if hasattr(f.rep, 'mul_ground'):
            result = f.rep.mul_ground(coeff)
        else:
            raise OperationNotSupported(f, 'mul_ground')
        return f.per(result)

    def quo_ground(f, coeff):
        """
        Quotient of ``f`` by a an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x + 4).quo_ground(2)
        Poly(x + 2, x, domain='ZZ')

        >>> Poly(2*x + 3).quo_ground(2)
        Poly(x + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'quo_ground'):
            result = f.rep.quo_ground(coeff)
        else:
            raise OperationNotSupported(f, 'quo_ground')
        return f.per(result)

    def exquo_ground(f, coeff):
        """
        Exact quotient of ``f`` by a an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x + 4).exquo_ground(2)
        Poly(x + 2, x, domain='ZZ')

        >>> Poly(2*x + 3).exquo_ground(2)
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2 does not divide 3 in ZZ

        """
        if hasattr(f.rep, 'exquo_ground'):
            result = f.rep.exquo_ground(coeff)
        else:
            raise OperationNotSupported(f, 'exquo_ground')
        return f.per(result)

    def abs(f):
        """
        Make all coefficients in ``f`` positive.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).abs()
        Poly(x**2 + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'abs'):
            result = f.rep.abs()
        else:
            raise OperationNotSupported(f, 'abs')
        return f.per(result)

    def neg(f):
        """
        Negate all coefficients in ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).neg()
        Poly(-x**2 + 1, x, domain='ZZ')

        >>> -Poly(x**2 - 1, x)
        Poly(-x**2 + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'neg'):
            result = f.rep.neg()
        else:
            raise OperationNotSupported(f, 'neg')
        return f.per(result)

    def add(f, g):
        """
        Add two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).add(Poly(x - 2, x))
        Poly(x**2 + x - 1, x, domain='ZZ')

        >>> Poly(x**2 + 1, x) + Poly(x - 2, x)
        Poly(x**2 + x - 1, x, domain='ZZ')

        """
        g = sympify(g)
        if not g.is_Poly:
            return f.add_ground(g)
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'add'):
            result = F.add(G)
        else:
            raise OperationNotSupported(f, 'add')
        return per(result)

    def sub(f, g):
        """
        Subtract two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).sub(Poly(x - 2, x))
        Poly(x**2 - x + 3, x, domain='ZZ')

        >>> Poly(x**2 + 1, x) - Poly(x - 2, x)
        Poly(x**2 - x + 3, x, domain='ZZ')

        """
        g = sympify(g)
        if not g.is_Poly:
            return f.sub_ground(g)
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'sub'):
            result = F.sub(G)
        else:
            raise OperationNotSupported(f, 'sub')
        return per(result)

    def mul(f, g):
        """
        Multiply two polynomials ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).mul(Poly(x - 2, x))
        Poly(x**3 - 2*x**2 + x - 2, x, domain='ZZ')

        >>> Poly(x**2 + 1, x)*Poly(x - 2, x)
        Poly(x**3 - 2*x**2 + x - 2, x, domain='ZZ')

        """
        g = sympify(g)
        if not g.is_Poly:
            return f.mul_ground(g)
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'mul'):
            result = F.mul(G)
        else:
            raise OperationNotSupported(f, 'mul')
        return per(result)

    def sqr(f):
        """
        Square a polynomial ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x - 2, x).sqr()
        Poly(x**2 - 4*x + 4, x, domain='ZZ')

        >>> Poly(x - 2, x)**2
        Poly(x**2 - 4*x + 4, x, domain='ZZ')

        """
        if hasattr(f.rep, 'sqr'):
            result = f.rep.sqr()
        else:
            raise OperationNotSupported(f, 'sqr')
        return f.per(result)

    def pow(f, n):
        """
        Raise ``f`` to a non-negative power ``n``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x - 2, x).pow(3)
        Poly(x**3 - 6*x**2 + 12*x - 8, x, domain='ZZ')

        >>> Poly(x - 2, x)**3
        Poly(x**3 - 6*x**2 + 12*x - 8, x, domain='ZZ')

        """
        n = int(n)
        if hasattr(f.rep, 'pow'):
            result = f.rep.pow(n)
        else:
            raise OperationNotSupported(f, 'pow')
        return f.per(result)

    def pdiv(f, g):
        """
        Polynomial pseudo-division of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).pdiv(Poly(2*x - 4, x))
        (Poly(2*x + 4, x, domain='ZZ'), Poly(20, x, domain='ZZ'))

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'pdiv'):
            q, r = F.pdiv(G)
        else:
            raise OperationNotSupported(f, 'pdiv')
        return (per(q), per(r))

    def prem(f, g):
        """
        Polynomial pseudo-remainder of ``f`` by ``g``.

        Caveat: The function prem(f, g, x) can be safely used to compute
          in Z[x] _only_ subresultant polynomial remainder sequences (prs's).

          To safely compute Euclidean and Sturmian prs's in Z[x]
          employ anyone of the corresponding functions found in
          the module sympy.polys.subresultants_qq_zz. The functions
          in the module with suffix _pg compute prs's in Z[x] employing
          rem(f, g, x), whereas the functions with suffix _amv
          compute prs's in Z[x] employing rem_z(f, g, x).

          The function rem_z(f, g, x) differs from prem(f, g, x) in that
          to compute the remainder polynomials in Z[x] it premultiplies
          the divident times the absolute value of the leading coefficient
          of the divisor raised to the power degree(f, x) - degree(g, x) + 1.


        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).prem(Poly(2*x - 4, x))
        Poly(20, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'prem'):
            result = F.prem(G)
        else:
            raise OperationNotSupported(f, 'prem')
        return per(result)

    def pquo(f, g):
        """
        Polynomial pseudo-quotient of ``f`` by ``g``.

        See the Caveat note in the function prem(f, g).

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).pquo(Poly(2*x - 4, x))
        Poly(2*x + 4, x, domain='ZZ')

        >>> Poly(x**2 - 1, x).pquo(Poly(2*x - 2, x))
        Poly(2*x + 2, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'pquo'):
            result = F.pquo(G)
        else:
            raise OperationNotSupported(f, 'pquo')
        return per(result)

    def pexquo(f, g):
        """
        Polynomial exact pseudo-quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).pexquo(Poly(2*x - 2, x))
        Poly(2*x + 2, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).pexquo(Poly(2*x - 4, x))
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'pexquo'):
            try:
                result = F.pexquo(G)
            except ExactQuotientFailed as exc:
                raise exc.new(f.as_expr(), g.as_expr())
        else:
            raise OperationNotSupported(f, 'pexquo')
        return per(result)

    def div(f, g, auto=True):
        """
        Polynomial division with remainder of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).div(Poly(2*x - 4, x))
        (Poly(1/2*x + 1, x, domain='QQ'), Poly(5, x, domain='QQ'))

        >>> Poly(x**2 + 1, x).div(Poly(2*x - 4, x), auto=False)
        (Poly(0, x, domain='ZZ'), Poly(x**2 + 1, x, domain='ZZ'))

        """
        dom, per, F, G = f._unify(g)
        retract = False
        if auto and dom.is_Ring and (not dom.is_Field):
            F, G = (F.to_field(), G.to_field())
            retract = True
        if hasattr(f.rep, 'div'):
            q, r = F.div(G)
        else:
            raise OperationNotSupported(f, 'div')
        if retract:
            try:
                Q, R = (q.to_ring(), r.to_ring())
            except CoercionFailed:
                pass
            else:
                q, r = (Q, R)
        return (per(q), per(r))

    def rem(f, g, auto=True):
        """
        Computes the polynomial remainder of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).rem(Poly(2*x - 4, x))
        Poly(5, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).rem(Poly(2*x - 4, x), auto=False)
        Poly(x**2 + 1, x, domain='ZZ')

        """
        dom, per, F, G = f._unify(g)
        retract = False
        if auto and dom.is_Ring and (not dom.is_Field):
            F, G = (F.to_field(), G.to_field())
            retract = True
        if hasattr(f.rep, 'rem'):
            r = F.rem(G)
        else:
            raise OperationNotSupported(f, 'rem')
        if retract:
            try:
                r = r.to_ring()
            except CoercionFailed:
                pass
        return per(r)

    def quo(f, g, auto=True):
        """
        Computes polynomial quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).quo(Poly(2*x - 4, x))
        Poly(1/2*x + 1, x, domain='QQ')

        >>> Poly(x**2 - 1, x).quo(Poly(x - 1, x))
        Poly(x + 1, x, domain='ZZ')

        """
        dom, per, F, G = f._unify(g)
        retract = False
        if auto and dom.is_Ring and (not dom.is_Field):
            F, G = (F.to_field(), G.to_field())
            retract = True
        if hasattr(f.rep, 'quo'):
            q = F.quo(G)
        else:
            raise OperationNotSupported(f, 'quo')
        if retract:
            try:
                q = q.to_ring()
            except CoercionFailed:
                pass
        return per(q)

    def exquo(f, g, auto=True):
        """
        Computes polynomial exact quotient of ``f`` by ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).exquo(Poly(x - 1, x))
        Poly(x + 1, x, domain='ZZ')

        >>> Poly(x**2 + 1, x).exquo(Poly(2*x - 4, x))
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1

        """
        dom, per, F, G = f._unify(g)
        retract = False
        if auto and dom.is_Ring and (not dom.is_Field):
            F, G = (F.to_field(), G.to_field())
            retract = True
        if hasattr(f.rep, 'exquo'):
            try:
                q = F.exquo(G)
            except ExactQuotientFailed as exc:
                raise exc.new(f.as_expr(), g.as_expr())
        else:
            raise OperationNotSupported(f, 'exquo')
        if retract:
            try:
                q = q.to_ring()
            except CoercionFailed:
                pass
        return per(q)

    def _gen_to_level(f, gen):
        """Returns level associated with the given generator. """
        if isinstance(gen, int):
            length = len(f.gens)
            if -length <= gen < length:
                if gen < 0:
                    return length + gen
                else:
                    return gen
            else:
                raise PolynomialError('-%s <= gen < %s expected, got %s' % (length, length, gen))
        else:
            try:
                return f.gens.index(sympify(gen))
            except ValueError:
                raise PolynomialError('a valid generator expected, got %s' % gen)

    def degree(f, gen=0):
        """
        Returns degree of ``f`` in ``x_j``.

        The degree of 0 is negative infinity.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).degree()
        2
        >>> Poly(x**2 + y*x + y, x, y).degree(y)
        1
        >>> Poly(0, x).degree()
        -oo

        """
        j = f._gen_to_level(gen)
        if hasattr(f.rep, 'degree'):
            return f.rep.degree(j)
        else:
            raise OperationNotSupported(f, 'degree')

    def degree_list(f):
        """
        Returns a list of degrees of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).degree_list()
        (2, 1)

        """
        if hasattr(f.rep, 'degree_list'):
            return f.rep.degree_list()
        else:
            raise OperationNotSupported(f, 'degree_list')

    def total_degree(f):
        """
        Returns the total degree of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + y*x + 1, x, y).total_degree()
        2
        >>> Poly(x + y**5, x, y).total_degree()
        5

        """
        if hasattr(f.rep, 'total_degree'):
            return f.rep.total_degree()
        else:
            raise OperationNotSupported(f, 'total_degree')

    def homogenize(f, s):
        """
        Returns the homogeneous polynomial of ``f``.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. If you only
        want to check if a polynomial is homogeneous, then use
        :func:`Poly.is_homogeneous`. If you want not only to check if a
        polynomial is homogeneous but also compute its homogeneous order,
        then use :func:`Poly.homogeneous_order`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> f = Poly(x**5 + 2*x**2*y**2 + 9*x*y**3)
        >>> f.homogenize(z)
        Poly(x**5 + 2*x**2*y**2*z + 9*x*y**3*z, x, y, z, domain='ZZ')

        """
        if not isinstance(s, Symbol):
            raise TypeError('``Symbol`` expected, got %s' % type(s))
        if s in f.gens:
            i = f.gens.index(s)
            gens = f.gens
        else:
            i = len(f.gens)
            gens = f.gens + (s,)
        if hasattr(f.rep, 'homogenize'):
            return f.per(f.rep.homogenize(i), gens=gens)
        raise OperationNotSupported(f, 'homogeneous_order')

    def homogeneous_order(f):
        """
        Returns the homogeneous order of ``f``.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. This degree is
        the homogeneous order of ``f``. If you only want to check if a
        polynomial is homogeneous, then use :func:`Poly.is_homogeneous`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**5 + 2*x**3*y**2 + 9*x*y**4)
        >>> f.homogeneous_order()
        5

        """
        if hasattr(f.rep, 'homogeneous_order'):
            return f.rep.homogeneous_order()
        else:
            raise OperationNotSupported(f, 'homogeneous_order')

    def LC(f, order=None):
        """
        Returns the leading coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(4*x**3 + 2*x**2 + 3*x, x).LC()
        4

        """
        if order is not None:
            return f.coeffs(order)[0]
        if hasattr(f.rep, 'LC'):
            result = f.rep.LC()
        else:
            raise OperationNotSupported(f, 'LC')
        return f.rep.dom.to_sympy(result)

    def TC(f):
        """
        Returns the trailing coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x**2 + 3*x, x).TC()
        0

        """
        if hasattr(f.rep, 'TC'):
            result = f.rep.TC()
        else:
            raise OperationNotSupported(f, 'TC')
        return f.rep.dom.to_sympy(result)

    def EC(f, order=None):
        """
        Returns the last non-zero coefficient of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 + 2*x**2 + 3*x, x).EC()
        3

        """
        if hasattr(f.rep, 'coeffs'):
            return f.coeffs(order)[-1]
        else:
            raise OperationNotSupported(f, 'EC')

    def coeff_monomial(f, monom):
        """
        Returns the coefficient of ``monom`` in ``f`` if there, else None.

        Examples
        ========

        >>> from sympy import Poly, exp
        >>> from sympy.abc import x, y

        >>> p = Poly(24*x*y*exp(8) + 23*x, x, y)

        >>> p.coeff_monomial(x)
        23
        >>> p.coeff_monomial(y)
        0
        >>> p.coeff_monomial(x*y)
        24*exp(8)

        Note that ``Expr.coeff()`` behaves differently, collecting terms
        if possible; the Poly must be converted to an Expr to use that
        method, however:

        >>> p.as_expr().coeff(x)
        24*y*exp(8) + 23
        >>> p.as_expr().coeff(y)
        24*x*exp(8)
        >>> p.as_expr().coeff(x*y)
        24*exp(8)

        See Also
        ========
        nth: more efficient query using exponents of the monomial's generators

        """
        return f.nth(*Monomial(monom, f.gens).exponents)

    def nth(f, *N):
        """
        Returns the ``n``-th coefficient of ``f`` where ``N`` are the
        exponents of the generators in the term of interest.

        Examples
        ========

        >>> from sympy import Poly, sqrt
        >>> from sympy.abc import x, y

        >>> Poly(x**3 + 2*x**2 + 3*x, x).nth(2)
        2
        >>> Poly(x**3 + 2*x*y**2 + y**2, x, y).nth(1, 2)
        2
        >>> Poly(4*sqrt(x)*y)
        Poly(4*y*(sqrt(x)), y, sqrt(x), domain='ZZ')
        >>> _.nth(1, 1)
        4

        See Also
        ========
        coeff_monomial

        """
        if hasattr(f.rep, 'nth'):
            if len(N) != len(f.gens):
                raise ValueError('exponent of each generator must be specified')
            result = f.rep.nth(*list(map(int, N)))
        else:
            raise OperationNotSupported(f, 'nth')
        return f.rep.dom.to_sympy(result)

    def coeff(f, x, n=1, right=False):
        raise NotImplementedError("Either convert to Expr with `as_expr` method to use Expr's coeff method or else use the `coeff_monomial` method of Polys.")

    def LM(f, order=None):
        """
        Returns the leading monomial of ``f``.

        The Leading monomial signifies the monomial having
        the highest power of the principal generator in the
        expression f.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).LM()
        x**2*y**0

        """
        return Monomial(f.monoms(order)[0], f.gens)

    def EM(f, order=None):
        """
        Returns the last non-zero monomial of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).EM()
        x**0*y**1

        """
        return Monomial(f.monoms(order)[-1], f.gens)

    def LT(f, order=None):
        """
        Returns the leading term of ``f``.

        The Leading term signifies the term having
        the highest power of the principal generator in the
        expression f along with its coefficient.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).LT()
        (x**2*y**0, 4)

        """
        monom, coeff = f.terms(order)[0]
        return (Monomial(monom, f.gens), coeff)

    def ET(f, order=None):
        """
        Returns the last non-zero term of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).ET()
        (x**0*y**1, 3)

        """
        monom, coeff = f.terms(order)[-1]
        return (Monomial(monom, f.gens), coeff)

    def max_norm(f):
        """
        Returns maximum norm of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(-x**2 + 2*x - 3, x).max_norm()
        3

        """
        if hasattr(f.rep, 'max_norm'):
            result = f.rep.max_norm()
        else:
            raise OperationNotSupported(f, 'max_norm')
        return f.rep.dom.to_sympy(result)

    def l1_norm(f):
        """
        Returns l1 norm of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(-x**2 + 2*x - 3, x).l1_norm()
        6

        """
        if hasattr(f.rep, 'l1_norm'):
            result = f.rep.l1_norm()
        else:
            raise OperationNotSupported(f, 'l1_norm')
        return f.rep.dom.to_sympy(result)

    def clear_denoms(self, convert=False):
        """
        Clear denominators, but keep the ground domain.

        Examples
        ========

        >>> from sympy import Poly, S, QQ
        >>> from sympy.abc import x

        >>> f = Poly(x/2 + S(1)/3, x, domain=QQ)

        >>> f.clear_denoms()
        (6, Poly(3*x + 2, x, domain='QQ'))
        >>> f.clear_denoms(convert=True)
        (6, Poly(3*x + 2, x, domain='ZZ'))

        """
        f = self
        if not f.rep.dom.is_Field:
            return (S.One, f)
        dom = f.get_domain()
        if dom.has_assoc_Ring:
            dom = f.rep.dom.get_ring()
        if hasattr(f.rep, 'clear_denoms'):
            coeff, result = f.rep.clear_denoms()
        else:
            raise OperationNotSupported(f, 'clear_denoms')
        coeff, f = (dom.to_sympy(coeff), f.per(result))
        if not convert or not dom.has_assoc_Ring:
            return (coeff, f)
        else:
            return (coeff, f.to_ring())

    def rat_clear_denoms(self, g):
        """
        Clear denominators in a rational function ``f/g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2/y + 1, x)
        >>> g = Poly(x**3 + y, x)

        >>> p, q = f.rat_clear_denoms(g)

        >>> p
        Poly(x**2 + y, x, domain='ZZ[y]')
        >>> q
        Poly(y*x**3 + y**2, x, domain='ZZ[y]')

        """
        f = self
        dom, per, f, g = f._unify(g)
        f = per(f)
        g = per(g)
        if not (dom.is_Field and dom.has_assoc_Ring):
            return (f, g)
        a, f = f.clear_denoms(convert=True)
        b, g = g.clear_denoms(convert=True)
        f = f.mul_ground(b)
        g = g.mul_ground(a)
        return (f, g)

    def integrate(self, *specs, **args):
        """
        Computes indefinite integral of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x + 1, x).integrate()
        Poly(1/3*x**3 + x**2 + x, x, domain='QQ')

        >>> Poly(x*y**2 + x, x, y).integrate((0, 1), (1, 0))
        Poly(1/2*x**2*y**2 + 1/2*x**2, x, y, domain='QQ')

        """
        f = self
        if args.get('auto', True) and f.rep.dom.is_Ring:
            f = f.to_field()
        if hasattr(f.rep, 'integrate'):
            if not specs:
                return f.per(f.rep.integrate(m=1))
            rep = f.rep
            for spec in specs:
                if isinstance(spec, tuple):
                    gen, m = spec
                else:
                    gen, m = (spec, 1)
                rep = rep.integrate(int(m), f._gen_to_level(gen))
            return f.per(rep)
        else:
            raise OperationNotSupported(f, 'integrate')

    def diff(f, *specs, **kwargs):
        """
        Computes partial derivative of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + 2*x + 1, x).diff()
        Poly(2*x + 2, x, domain='ZZ')

        >>> Poly(x*y**2 + x, x, y).diff((0, 0), (1, 1))
        Poly(2*x*y, x, y, domain='ZZ')

        """
        if not kwargs.get('evaluate', True):
            return Derivative(f, *specs, **kwargs)
        if hasattr(f.rep, 'diff'):
            if not specs:
                return f.per(f.rep.diff(m=1))
            rep = f.rep
            for spec in specs:
                if isinstance(spec, tuple):
                    gen, m = spec
                else:
                    gen, m = (spec, 1)
                rep = rep.diff(int(m), f._gen_to_level(gen))
            return f.per(rep)
        else:
            raise OperationNotSupported(f, 'diff')
    _eval_derivative = diff

    def eval(self, x, a=None, auto=True):
        """
        Evaluate ``f`` at ``a`` in the given variable.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> Poly(x**2 + 2*x + 3, x).eval(2)
        11

        >>> Poly(2*x*y + 3*x + y + 2, x, y).eval(x, 2)
        Poly(5*y + 8, y, domain='ZZ')

        >>> f = Poly(2*x*y + 3*x + y + 2*z, x, y, z)

        >>> f.eval({x: 2})
        Poly(5*y + 2*z + 6, y, z, domain='ZZ')
        >>> f.eval({x: 2, y: 5})
        Poly(2*z + 31, z, domain='ZZ')
        >>> f.eval({x: 2, y: 5, z: 7})
        45

        >>> f.eval((2, 5))
        Poly(2*z + 31, z, domain='ZZ')
        >>> f(2, 5)
        Poly(2*z + 31, z, domain='ZZ')

        """
        f = self
        if a is None:
            if isinstance(x, dict):
                mapping = x
                for gen, value in mapping.items():
                    f = f.eval(gen, value)
                return f
            elif isinstance(x, (tuple, list)):
                values = x
                if len(values) > len(f.gens):
                    raise ValueError('too many values provided')
                for gen, value in zip(f.gens, values):
                    f = f.eval(gen, value)
                return f
            else:
                j, a = (0, x)
        else:
            j = f._gen_to_level(x)
        if not hasattr(f.rep, 'eval'):
            raise OperationNotSupported(f, 'eval')
        try:
            result = f.rep.eval(a, j)
        except CoercionFailed:
            if not auto:
                raise DomainError('Cannot evaluate at %s in %s' % (a, f.rep.dom))
            else:
                a_domain, [a] = construct_domain([a])
                new_domain = f.get_domain().unify_with_symbols(a_domain, f.gens)
                f = f.set_domain(new_domain)
                a = new_domain.convert(a, a_domain)
                result = f.rep.eval(a, j)
        return f.per(result, remove=j)

    def __call__(f, *values):
        """
        Evaluate ``f`` at the give values.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y, z

        >>> f = Poly(2*x*y + 3*x + y + 2*z, x, y, z)

        >>> f(2)
        Poly(5*y + 2*z + 6, y, z, domain='ZZ')
        >>> f(2, 5)
        Poly(2*z + 31, z, domain='ZZ')
        >>> f(2, 5, 7)
        45

        """
        return f.eval(values)

    def half_gcdex(f, g, auto=True):
        """
        Half extended Euclidean algorithm of ``f`` and ``g``.

        Returns ``(s, h)`` such that ``h = gcd(f, g)`` and ``s*f = h (mod g)``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
        >>> g = x**3 + x**2 - 4*x - 4

        >>> Poly(f).half_gcdex(Poly(g))
        (Poly(-1/5*x + 3/5, x, domain='QQ'), Poly(x + 1, x, domain='QQ'))

        """
        dom, per, F, G = f._unify(g)
        if auto and dom.is_Ring:
            F, G = (F.to_field(), G.to_field())
        if hasattr(f.rep, 'half_gcdex'):
            s, h = F.half_gcdex(G)
        else:
            raise OperationNotSupported(f, 'half_gcdex')
        return (per(s), per(h))

    def gcdex(f, g, auto=True):
        """
        Extended Euclidean algorithm of ``f`` and ``g``.

        Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
        >>> g = x**3 + x**2 - 4*x - 4

        >>> Poly(f).gcdex(Poly(g))
        (Poly(-1/5*x + 3/5, x, domain='QQ'),
         Poly(1/5*x**2 - 6/5*x + 2, x, domain='QQ'),
         Poly(x + 1, x, domain='QQ'))

        """
        dom, per, F, G = f._unify(g)
        if auto and dom.is_Ring:
            F, G = (F.to_field(), G.to_field())
        if hasattr(f.rep, 'gcdex'):
            s, t, h = F.gcdex(G)
        else:
            raise OperationNotSupported(f, 'gcdex')
        return (per(s), per(t), per(h))

    def invert(f, g, auto=True):
        """
        Invert ``f`` modulo ``g`` when possible.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).invert(Poly(2*x - 1, x))
        Poly(-4/3, x, domain='QQ')

        >>> Poly(x**2 - 1, x).invert(Poly(x - 1, x))
        Traceback (most recent call last):
        ...
        NotInvertible: zero divisor

        """
        dom, per, F, G = f._unify(g)
        if auto and dom.is_Ring:
            F, G = (F.to_field(), G.to_field())
        if hasattr(f.rep, 'invert'):
            result = F.invert(G)
        else:
            raise OperationNotSupported(f, 'invert')
        return per(result)

    def revert(f, n):
        """
        Compute ``f**(-1)`` mod ``x**n``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(1, x).revert(2)
        Poly(1, x, domain='ZZ')

        >>> Poly(1 + x, x).revert(1)
        Poly(1, x, domain='ZZ')

        >>> Poly(x**2 - 2, x).revert(2)
        Traceback (most recent call last):
        ...
        NotReversible: only units are reversible in a ring

        >>> Poly(1/x, x).revert(1)
        Traceback (most recent call last):
        ...
        PolynomialError: 1/x contains an element of the generators set

        """
        if hasattr(f.rep, 'revert'):
            result = f.rep.revert(int(n))
        else:
            raise OperationNotSupported(f, 'revert')
        return f.per(result)

    def subresultants(f, g):
        """
        Computes the subresultant PRS of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 1, x).subresultants(Poly(x**2 - 1, x))
        [Poly(x**2 + 1, x, domain='ZZ'),
         Poly(x**2 - 1, x, domain='ZZ'),
         Poly(-2, x, domain='ZZ')]

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'subresultants'):
            result = F.subresultants(G)
        else:
            raise OperationNotSupported(f, 'subresultants')
        return list(map(per, result))

    def resultant(f, g, includePRS=False):
        """
        Computes the resultant of ``f`` and ``g`` via PRS.

        If includePRS=True, it includes the subresultant PRS in the result.
        Because the PRS is used to calculate the resultant, this is more
        efficient than calling :func:`subresultants` separately.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(x**2 + 1, x)

        >>> f.resultant(Poly(x**2 - 1, x))
        4
        >>> f.resultant(Poly(x**2 - 1, x), includePRS=True)
        (4, [Poly(x**2 + 1, x, domain='ZZ'), Poly(x**2 - 1, x, domain='ZZ'),
             Poly(-2, x, domain='ZZ')])

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'resultant'):
            if includePRS:
                result, R = F.resultant(G, includePRS=includePRS)
            else:
                result = F.resultant(G)
        else:
            raise OperationNotSupported(f, 'resultant')
        if includePRS:
            return (per(result, remove=0), list(map(per, R)))
        return per(result, remove=0)

    def discriminant(f):
        """
        Computes the discriminant of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + 2*x + 3, x).discriminant()
        -8

        """
        if hasattr(f.rep, 'discriminant'):
            result = f.rep.discriminant()
        else:
            raise OperationNotSupported(f, 'discriminant')
        return f.per(result, remove=0)

    def dispersionset(f, g=None):
        """Compute the *dispersion set* of two polynomials.

        For two polynomials `f(x)` and `g(x)` with `\\deg f > 0`
        and `\\deg g > 0` the dispersion set `\\operatorname{J}(f, g)` is defined as:

        .. math::
            \\operatorname{J}(f, g)
            & := \\{a \\in \\mathbb{N}_0 | \\gcd(f(x), g(x+a)) \\neq 1\\} \\\\
            &  = \\{a \\in \\mathbb{N}_0 | \\deg \\gcd(f(x), g(x+a)) \\geq 1\\}

        For a single polynomial one defines `\\operatorname{J}(f) := \\operatorname{J}(f, f)`.

        Examples
        ========

        >>> from sympy import poly
        >>> from sympy.polys.dispersion import dispersion, dispersionset
        >>> from sympy.abc import x

        Dispersion set and dispersion of a simple polynomial:

        >>> fp = poly((x - 3)*(x + 3), x)
        >>> sorted(dispersionset(fp))
        [0, 6]
        >>> dispersion(fp)
        6

        Note that the definition of the dispersion is not symmetric:

        >>> fp = poly(x**4 - 3*x**2 + 1, x)
        >>> gp = fp.shift(-3)
        >>> sorted(dispersionset(fp, gp))
        [2, 3, 4]
        >>> dispersion(fp, gp)
        4
        >>> sorted(dispersionset(gp, fp))
        []
        >>> dispersion(gp, fp)
        -oo

        Computing the dispersion also works over field extensions:

        >>> from sympy import sqrt
        >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
        >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
        >>> sorted(dispersionset(fp, gp))
        [2]
        >>> sorted(dispersionset(gp, fp))
        [1, 4]

        We can even perform the computations for polynomials
        having symbolic coefficients:

        >>> from sympy.abc import a
        >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
        >>> sorted(dispersionset(fp))
        [0, 1]

        See Also
        ========

        dispersion

        References
        ==========

        1. [ManWright94]_
        2. [Koepf98]_
        3. [Abramov71]_
        4. [Man93]_
        """
        from sympy.polys.dispersion import dispersionset
        return dispersionset(f, g)

    def dispersion(f, g=None):
        """Compute the *dispersion* of polynomials.

        For two polynomials `f(x)` and `g(x)` with `\\deg f > 0`
        and `\\deg g > 0` the dispersion `\\operatorname{dis}(f, g)` is defined as:

        .. math::
            \\operatorname{dis}(f, g)
            & := \\max\\{ J(f,g) \\cup \\{0\\} \\} \\\\
            &  = \\max\\{ \\{a \\in \\mathbb{N} | \\gcd(f(x), g(x+a)) \\neq 1\\} \\cup \\{0\\} \\}

        and for a single polynomial `\\operatorname{dis}(f) := \\operatorname{dis}(f, f)`.

        Examples
        ========

        >>> from sympy import poly
        >>> from sympy.polys.dispersion import dispersion, dispersionset
        >>> from sympy.abc import x

        Dispersion set and dispersion of a simple polynomial:

        >>> fp = poly((x - 3)*(x + 3), x)
        >>> sorted(dispersionset(fp))
        [0, 6]
        >>> dispersion(fp)
        6

        Note that the definition of the dispersion is not symmetric:

        >>> fp = poly(x**4 - 3*x**2 + 1, x)
        >>> gp = fp.shift(-3)
        >>> sorted(dispersionset(fp, gp))
        [2, 3, 4]
        >>> dispersion(fp, gp)
        4
        >>> sorted(dispersionset(gp, fp))
        []
        >>> dispersion(gp, fp)
        -oo

        Computing the dispersion also works over field extensions:

        >>> from sympy import sqrt
        >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
        >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
        >>> sorted(dispersionset(fp, gp))
        [2]
        >>> sorted(dispersionset(gp, fp))
        [1, 4]

        We can even perform the computations for polynomials
        having symbolic coefficients:

        >>> from sympy.abc import a
        >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
        >>> sorted(dispersionset(fp))
        [0, 1]

        See Also
        ========

        dispersionset

        References
        ==========

        1. [ManWright94]_
        2. [Koepf98]_
        3. [Abramov71]_
        4. [Man93]_
        """
        from sympy.polys.dispersion import dispersion
        return dispersion(f, g)

    def cofactors(f, g):
        """
        Returns the GCD of ``f`` and ``g`` and their cofactors.

        Returns polynomials ``(h, cff, cfg)`` such that ``h = gcd(f, g)``, and
        ``cff = quo(f, h)`` and ``cfg = quo(g, h)`` are, so called, cofactors
        of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).cofactors(Poly(x**2 - 3*x + 2, x))
        (Poly(x - 1, x, domain='ZZ'),
         Poly(x + 1, x, domain='ZZ'),
         Poly(x - 2, x, domain='ZZ'))

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'cofactors'):
            h, cff, cfg = F.cofactors(G)
        else:
            raise OperationNotSupported(f, 'cofactors')
        return (per(h), per(cff), per(cfg))

    def gcd(f, g):
        """
        Returns the polynomial GCD of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).gcd(Poly(x**2 - 3*x + 2, x))
        Poly(x - 1, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'gcd'):
            result = F.gcd(G)
        else:
            raise OperationNotSupported(f, 'gcd')
        return per(result)

    def lcm(f, g):
        """
        Returns polynomial LCM of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 1, x).lcm(Poly(x**2 - 3*x + 2, x))
        Poly(x**3 - 2*x**2 - x + 2, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'lcm'):
            result = F.lcm(G)
        else:
            raise OperationNotSupported(f, 'lcm')
        return per(result)

    def trunc(f, p):
        """
        Reduce ``f`` modulo a constant ``p``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**3 + 3*x**2 + 5*x + 7, x).trunc(3)
        Poly(-x**3 - x + 1, x, domain='ZZ')

        """
        p = f.rep.dom.convert(p)
        if hasattr(f.rep, 'trunc'):
            result = f.rep.trunc(p)
        else:
            raise OperationNotSupported(f, 'trunc')
        return f.per(result)

    def monic(self, auto=True):
        """
        Divides all coefficients by ``LC(f)``.

        Examples
        ========

        >>> from sympy import Poly, ZZ
        >>> from sympy.abc import x

        >>> Poly(3*x**2 + 6*x + 9, x, domain=ZZ).monic()
        Poly(x**2 + 2*x + 3, x, domain='QQ')

        >>> Poly(3*x**2 + 4*x + 2, x, domain=ZZ).monic()
        Poly(x**2 + 4/3*x + 2/3, x, domain='QQ')

        """
        f = self
        if auto and f.rep.dom.is_Ring:
            f = f.to_field()
        if hasattr(f.rep, 'monic'):
            result = f.rep.monic()
        else:
            raise OperationNotSupported(f, 'monic')
        return f.per(result)

    def content(f):
        """
        Returns the GCD of polynomial coefficients.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(6*x**2 + 8*x + 12, x).content()
        2

        """
        if hasattr(f.rep, 'content'):
            result = f.rep.content()
        else:
            raise OperationNotSupported(f, 'content')
        return f.rep.dom.to_sympy(result)

    def primitive(f):
        """
        Returns the content and a primitive form of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**2 + 8*x + 12, x).primitive()
        (2, Poly(x**2 + 4*x + 6, x, domain='ZZ'))

        """
        if hasattr(f.rep, 'primitive'):
            cont, result = f.rep.primitive()
        else:
            raise OperationNotSupported(f, 'primitive')
        return (f.rep.dom.to_sympy(cont), f.per(result))

    def compose(f, g):
        """
        Computes the functional composition of ``f`` and ``g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + x, x).compose(Poly(x - 1, x))
        Poly(x**2 - x, x, domain='ZZ')

        """
        _, per, F, G = f._unify(g)
        if hasattr(f.rep, 'compose'):
            result = F.compose(G)
        else:
            raise OperationNotSupported(f, 'compose')
        return per(result)

    def decompose(f):
        """
        Computes a functional decomposition of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**4 + 2*x**3 - x - 1, x, domain='ZZ').decompose()
        [Poly(x**2 - x - 1, x, domain='ZZ'), Poly(x**2 + x, x, domain='ZZ')]

        """
        if hasattr(f.rep, 'decompose'):
            result = f.rep.decompose()
        else:
            raise OperationNotSupported(f, 'decompose')
        return list(map(f.per, result))

    def shift(f, a):
        """
        Efficiently compute Taylor shift ``f(x + a)``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 2*x + 1, x).shift(2)
        Poly(x**2 + 2*x + 1, x, domain='ZZ')

        """
        if hasattr(f.rep, 'shift'):
            result = f.rep.shift(a)
        else:
            raise OperationNotSupported(f, 'shift')
        return f.per(result)

    def transform(f, p, q):
        """
        Efficiently evaluate the functional transformation ``q**n * f(p/q)``.


        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1, x), Poly(x - 1, x))
        Poly(4, x, domain='ZZ')

        """
        P, Q = p.unify(q)
        F, P = f.unify(P)
        F, Q = F.unify(Q)
        if hasattr(F.rep, 'transform'):
            result = F.rep.transform(P.rep, Q.rep)
        else:
            raise OperationNotSupported(F, 'transform')
        return F.per(result)

    def sturm(self, auto=True):
        """
        Computes the Sturm sequence of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 - 2*x**2 + x - 3, x).sturm()
        [Poly(x**3 - 2*x**2 + x - 3, x, domain='QQ'),
         Poly(3*x**2 - 4*x + 1, x, domain='QQ'),
         Poly(2/9*x + 25/9, x, domain='QQ'),
         Poly(-2079/4, x, domain='QQ')]

        """
        f = self
        if auto and f.rep.dom.is_Ring:
            f = f.to_field()
        if hasattr(f.rep, 'sturm'):
            result = f.rep.sturm()
        else:
            raise OperationNotSupported(f, 'sturm')
        return list(map(f.per, result))

    def gff_list(f):
        """
        Computes greatest factorial factorization of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**5 + 2*x**4 - x**3 - 2*x**2

        >>> Poly(f).gff_list()
        [(Poly(x, x, domain='ZZ'), 1), (Poly(x + 2, x, domain='ZZ'), 4)]

        """
        if hasattr(f.rep, 'gff_list'):
            result = f.rep.gff_list()
        else:
            raise OperationNotSupported(f, 'gff_list')
        return [(f.per(g), k) for g, k in result]

    def norm(f):
        """
        Computes the product, ``Norm(f)``, of the conjugates of
        a polynomial ``f`` defined over a number field ``K``.

        Examples
        ========

        >>> from sympy import Poly, sqrt
        >>> from sympy.abc import x

        >>> a, b = sqrt(2), sqrt(3)

        A polynomial over a quadratic extension.
        Two conjugates x - a and x + a.

        >>> f = Poly(x - a, x, extension=a)
        >>> f.norm()
        Poly(x**2 - 2, x, domain='QQ')

        A polynomial over a quartic extension.
        Four conjugates x - a, x - a, x + a and x + a.

        >>> f = Poly(x - a, x, extension=(a, b))
        >>> f.norm()
        Poly(x**4 - 4*x**2 + 4, x, domain='QQ')

        """
        if hasattr(f.rep, 'norm'):
            r = f.rep.norm()
        else:
            raise OperationNotSupported(f, 'norm')
        return f.per(r)

    def sqf_norm(f):
        """
        Computes square-free norm of ``f``.

        Returns ``s``, ``f``, ``r``, such that ``g(x) = f(x-sa)`` and
        ``r(x) = Norm(g(x))`` is a square-free polynomial over ``K``,
        where ``a`` is the algebraic extension of the ground domain.

        Examples
        ========

        >>> from sympy import Poly, sqrt
        >>> from sympy.abc import x

        >>> s, f, r = Poly(x**2 + 1, x, extension=[sqrt(3)]).sqf_norm()

        >>> s
        1
        >>> f
        Poly(x**2 - 2*sqrt(3)*x + 4, x, domain='QQ<sqrt(3)>')
        >>> r
        Poly(x**4 - 4*x**2 + 16, x, domain='QQ')

        """
        if hasattr(f.rep, 'sqf_norm'):
            s, g, r = f.rep.sqf_norm()
        else:
            raise OperationNotSupported(f, 'sqf_norm')
        return (s, f.per(g), f.per(r))

    def sqf_part(f):
        """
        Computes square-free part of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**3 - 3*x - 2, x).sqf_part()
        Poly(x**2 - x - 2, x, domain='ZZ')

        """
        if hasattr(f.rep, 'sqf_part'):
            result = f.rep.sqf_part()
        else:
            raise OperationNotSupported(f, 'sqf_part')
        return f.per(result)

    def sqf_list(f, all=False):
        """
        Returns a list of square-free factors of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

        >>> Poly(f).sqf_list()
        (2, [(Poly(x + 1, x, domain='ZZ'), 2),
             (Poly(x + 2, x, domain='ZZ'), 3)])

        >>> Poly(f).sqf_list(all=True)
        (2, [(Poly(1, x, domain='ZZ'), 1),
             (Poly(x + 1, x, domain='ZZ'), 2),
             (Poly(x + 2, x, domain='ZZ'), 3)])

        """
        if hasattr(f.rep, 'sqf_list'):
            coeff, factors = f.rep.sqf_list(all)
        else:
            raise OperationNotSupported(f, 'sqf_list')
        return (f.rep.dom.to_sympy(coeff), [(f.per(g), k) for g, k in factors])

    def sqf_list_include(f, all=False):
        """
        Returns a list of square-free factors of ``f``.

        Examples
        ========

        >>> from sympy import Poly, expand
        >>> from sympy.abc import x

        >>> f = expand(2*(x + 1)**3*x**4)
        >>> f
        2*x**7 + 6*x**6 + 6*x**5 + 2*x**4

        >>> Poly(f).sqf_list_include()
        [(Poly(2, x, domain='ZZ'), 1),
         (Poly(x + 1, x, domain='ZZ'), 3),
         (Poly(x, x, domain='ZZ'), 4)]

        >>> Poly(f).sqf_list_include(all=True)
        [(Poly(2, x, domain='ZZ'), 1),
         (Poly(1, x, domain='ZZ'), 2),
         (Poly(x + 1, x, domain='ZZ'), 3),
         (Poly(x, x, domain='ZZ'), 4)]

        """
        if hasattr(f.rep, 'sqf_list_include'):
            factors = f.rep.sqf_list_include(all)
        else:
            raise OperationNotSupported(f, 'sqf_list_include')
        return [(f.per(g), k) for g, k in factors]

    def factor_list(f):
        """
        Returns a list of irreducible factors of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = 2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y

        >>> Poly(f).factor_list()
        (2, [(Poly(x + y, x, y, domain='ZZ'), 1),
             (Poly(x**2 + 1, x, y, domain='ZZ'), 2)])

        """
        if hasattr(f.rep, 'factor_list'):
            try:
                coeff, factors = f.rep.factor_list()
            except DomainError:
                return (S.One, [(f, 1)])
        else:
            raise OperationNotSupported(f, 'factor_list')
        return (f.rep.dom.to_sympy(coeff), [(f.per(g), k) for g, k in factors])

    def factor_list_include(f):
        """
        Returns a list of irreducible factors of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = 2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y

        >>> Poly(f).factor_list_include()
        [(Poly(2*x + 2*y, x, y, domain='ZZ'), 1),
         (Poly(x**2 + 1, x, y, domain='ZZ'), 2)]

        """
        if hasattr(f.rep, 'factor_list_include'):
            try:
                factors = f.rep.factor_list_include()
            except DomainError:
                return [(f, 1)]
        else:
            raise OperationNotSupported(f, 'factor_list_include')
        return [(f.per(g), k) for g, k in factors]

    def intervals(f, all=False, eps=None, inf=None, sup=None, fast=False, sqf=False):
        """
        Compute isolating intervals for roots of ``f``.

        For real roots the Vincent-Akritas-Strzebonski (VAS) continued fractions method is used.

        References
        ==========
        .. [#] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative Study of Two Real Root
            Isolation Methods . Nonlinear Analysis: Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
        .. [#] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S. Vigklas: Improving the
            Performance of the Continued Fractions Method Using new Bounds of Positive Roots. Nonlinear
            Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 3, x).intervals()
        [((-2, -1), 1), ((1, 2), 1)]
        >>> Poly(x**2 - 3, x).intervals(eps=1e-2)
        [((-26/15, -19/11), 1), ((19/11, 26/15), 1)]

        """
        if eps is not None:
            eps = QQ.convert(eps)
            if eps <= 0:
                raise ValueError("'eps' must be a positive rational")
        if inf is not None:
            inf = QQ.convert(inf)
        if sup is not None:
            sup = QQ.convert(sup)
        if hasattr(f.rep, 'intervals'):
            result = f.rep.intervals(all=all, eps=eps, inf=inf, sup=sup, fast=fast, sqf=sqf)
        else:
            raise OperationNotSupported(f, 'intervals')
        if sqf:

            def _real(interval):
                s, t = interval
                return (QQ.to_sympy(s), QQ.to_sympy(t))
            if not all:
                return list(map(_real, result))

            def _complex(rectangle):
                (u, v), (s, t) = rectangle
                return (QQ.to_sympy(u) + I * QQ.to_sympy(v), QQ.to_sympy(s) + I * QQ.to_sympy(t))
            real_part, complex_part = result
            return (list(map(_real, real_part)), list(map(_complex, complex_part)))
        else:

            def _real(interval):
                (s, t), k = interval
                return ((QQ.to_sympy(s), QQ.to_sympy(t)), k)
            if not all:
                return list(map(_real, result))

            def _complex(rectangle):
                ((u, v), (s, t)), k = rectangle
                return ((QQ.to_sympy(u) + I * QQ.to_sympy(v), QQ.to_sympy(s) + I * QQ.to_sympy(t)), k)
            real_part, complex_part = result
            return (list(map(_real, real_part)), list(map(_complex, complex_part)))

    def refine_root(f, s, t, eps=None, steps=None, fast=False, check_sqf=False):
        """
        Refine an isolating interval of a root to the given precision.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 3, x).refine_root(1, 2, eps=1e-2)
        (19/11, 26/15)

        """
        if check_sqf and (not f.is_sqf):
            raise PolynomialError('only square-free polynomials supported')
        s, t = (QQ.convert(s), QQ.convert(t))
        if eps is not None:
            eps = QQ.convert(eps)
            if eps <= 0:
                raise ValueError("'eps' must be a positive rational")
        if steps is not None:
            steps = int(steps)
        elif eps is None:
            steps = 1
        if hasattr(f.rep, 'refine_root'):
            S, T = f.rep.refine_root(s, t, eps=eps, steps=steps, fast=fast)
        else:
            raise OperationNotSupported(f, 'refine_root')
        return (QQ.to_sympy(S), QQ.to_sympy(T))

    def count_roots(f, inf=None, sup=None):
        """
        Return the number of roots of ``f`` in ``[inf, sup]`` interval.

        Examples
        ========

        >>> from sympy import Poly, I
        >>> from sympy.abc import x

        >>> Poly(x**4 - 4, x).count_roots(-3, 3)
        2
        >>> Poly(x**4 - 4, x).count_roots(0, 1 + 3*I)
        1

        """
        inf_real, sup_real = (True, True)
        if inf is not None:
            inf = sympify(inf)
            if inf is S.NegativeInfinity:
                inf = None
            else:
                re, im = inf.as_real_imag()
                if not im:
                    inf = QQ.convert(inf)
                else:
                    inf, inf_real = (list(map(QQ.convert, (re, im))), False)
        if sup is not None:
            sup = sympify(sup)
            if sup is S.Infinity:
                sup = None
            else:
                re, im = sup.as_real_imag()
                if not im:
                    sup = QQ.convert(sup)
                else:
                    sup, sup_real = (list(map(QQ.convert, (re, im))), False)
        if inf_real and sup_real:
            if hasattr(f.rep, 'count_real_roots'):
                count = f.rep.count_real_roots(inf=inf, sup=sup)
            else:
                raise OperationNotSupported(f, 'count_real_roots')
        else:
            if inf_real and inf is not None:
                inf = (inf, QQ.zero)
            if sup_real and sup is not None:
                sup = (sup, QQ.zero)
            if hasattr(f.rep, 'count_complex_roots'):
                count = f.rep.count_complex_roots(inf=inf, sup=sup)
            else:
                raise OperationNotSupported(f, 'count_complex_roots')
        return Integer(count)

    def root(f, index, radicals=True):
        """
        Get an indexed root of a polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(2*x**3 - 7*x**2 + 4*x + 4)

        >>> f.root(0)
        -1/2
        >>> f.root(1)
        2
        >>> f.root(2)
        2
        >>> f.root(3)
        Traceback (most recent call last):
        ...
        IndexError: root index out of [-3, 2] range, got 3

        >>> Poly(x**5 + x + 1).root(0)
        CRootOf(x**3 - x**2 + 1, 0)

        """
        return sympy.polys.rootoftools.rootof(f, index, radicals=radicals)

    def real_roots(f, multiple=True, radicals=True):
        """
        Return a list of real roots with multiplicities.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**3 - 7*x**2 + 4*x + 4).real_roots()
        [-1/2, 2, 2]
        >>> Poly(x**3 + x + 1).real_roots()
        [CRootOf(x**3 + x + 1, 0)]

        """
        reals = sympy.polys.rootoftools.CRootOf.real_roots(f, radicals=radicals)
        if multiple:
            return reals
        else:
            return group(reals, multiple=False)

    def all_roots(f, multiple=True, radicals=True):
        """
        Return a list of real and complex roots with multiplicities.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**3 - 7*x**2 + 4*x + 4).all_roots()
        [-1/2, 2, 2]
        >>> Poly(x**3 + x + 1).all_roots()
        [CRootOf(x**3 + x + 1, 0),
         CRootOf(x**3 + x + 1, 1),
         CRootOf(x**3 + x + 1, 2)]

        """
        roots = sympy.polys.rootoftools.CRootOf.all_roots(f, radicals=radicals)
        if multiple:
            return roots
        else:
            return group(roots, multiple=False)

    def nroots(f, n=15, maxsteps=50, cleanup=True):
        """
        Compute numerical approximations of roots of ``f``.

        Parameters
        ==========

        n ... the number of digits to calculate
        maxsteps ... the maximum number of iterations to do

        If the accuracy `n` cannot be reached in `maxsteps`, it will raise an
        exception. You need to rerun with higher maxsteps.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 3).nroots(n=15)
        [-1.73205080756888, 1.73205080756888]
        >>> Poly(x**2 - 3).nroots(n=30)
        [-1.73205080756887729352744634151, 1.73205080756887729352744634151]

        """
        if f.is_multivariate:
            raise MultivariatePolynomialError('Cannot compute numerical roots of %s' % f)
        if f.degree() <= 0:
            return []
        if f.rep.dom is ZZ:
            coeffs = [int(coeff) for coeff in f.all_coeffs()]
        elif f.rep.dom is QQ:
            denoms = [coeff.q for coeff in f.all_coeffs()]
            fac = ilcm(*denoms)
            coeffs = [int(coeff * fac) for coeff in f.all_coeffs()]
        else:
            coeffs = [coeff.evalf(n=n).as_real_imag() for coeff in f.all_coeffs()]
            try:
                coeffs = [mpmath.mpc(*coeff) for coeff in coeffs]
            except TypeError:
                raise DomainError('Numerical domain expected, got %s' % f.rep.dom)
        dps = mpmath.mp.dps
        mpmath.mp.dps = n
        from sympy.functions.elementary.complexes import sign
        try:
            roots = mpmath.polyroots(coeffs, maxsteps=maxsteps, cleanup=cleanup, error=False, extraprec=f.degree() * 10)
            roots = list(map(sympify, sorted(roots, key=lambda r: (1 if r.imag else 0, r.real, abs(r.imag), sign(r.imag)))))
        except NoConvergence:
            try:
                roots = mpmath.polyroots(coeffs, maxsteps=maxsteps, cleanup=cleanup, error=False, extraprec=f.degree() * 15)
                roots = list(map(sympify, sorted(roots, key=lambda r: (1 if r.imag else 0, r.real, abs(r.imag), sign(r.imag)))))
            except NoConvergence:
                raise NoConvergence('convergence to root failed; try n < %s or maxsteps > %s' % (n, maxsteps))
        finally:
            mpmath.mp.dps = dps
        return roots

    def ground_roots(f):
        """
        Compute roots of ``f`` by factorization in the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**6 - 4*x**4 + 4*x**3 - x**2).ground_roots()
        {0: 2, 1: 2}

        """
        if f.is_multivariate:
            raise MultivariatePolynomialError('Cannot compute ground roots of %s' % f)
        roots = {}
        for factor, k in f.factor_list()[1]:
            if factor.is_linear:
                a, b = factor.all_coeffs()
                roots[-b / a] = k
        return roots

    def nth_power_roots_poly(f, n):
        """
        Construct a polynomial with n-th powers of roots of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(x**4 - x**2 + 1)

        >>> f.nth_power_roots_poly(2)
        Poly(x**4 - 2*x**3 + 3*x**2 - 2*x + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(3)
        Poly(x**4 + 2*x**2 + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(4)
        Poly(x**4 + 2*x**3 + 3*x**2 + 2*x + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(12)
        Poly(x**4 - 4*x**3 + 6*x**2 - 4*x + 1, x, domain='ZZ')

        """
        if f.is_multivariate:
            raise MultivariatePolynomialError('must be a univariate polynomial')
        N = sympify(n)
        if N.is_Integer and N >= 1:
            n = int(N)
        else:
            raise ValueError("'n' must an integer and n >= 1, got %s" % n)
        x = f.gen
        t = Dummy('t')
        r = f.resultant(f.__class__.from_expr(x ** n - t, x, t))
        return r.replace(t, x)

    def same_root(f, a, b):
        """
        Decide whether two roots of this polynomial are equal.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly, exp, I, pi
        >>> f = Poly(cyclotomic_poly(5))
        >>> r0 = exp(2*I*pi/5)
        >>> indices = [i for i, r in enumerate(f.all_roots()) if f.same_root(r, r0)]
        >>> print(indices)
        [3]

        Raises
        ======

        DomainError
            If the domain of the polynomial is not :ref:`ZZ`, :ref:`QQ`,
            :ref:`RR`, or :ref:`CC`.
        MultivariatePolynomialError
            If the polynomial is not univariate.
        PolynomialError
            If the polynomial is of degree < 2.

        """
        if f.is_multivariate:
            raise MultivariatePolynomialError('Must be a univariate polynomial')
        dom_delta_sq = f.rep.mignotte_sep_bound_squared()
        delta_sq = f.domain.get_field().to_sympy(dom_delta_sq)
        eps_sq = delta_sq / 9
        r, _, _, _ = evalf(1 / eps_sq, 1, {})
        n = fastlog(r)
        m = n // 2 + n % 2
        ev = lambda x: quad_to_mpmath(_evalf_with_bounded_error(x, m=m))
        A, B = (ev(a), ev(b))
        return (A.real - B.real) ** 2 + (A.imag - B.imag) ** 2 < eps_sq

    def cancel(f, g, include=False):
        """
        Cancel common factors in a rational function ``f/g``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**2 - 2, x).cancel(Poly(x**2 - 2*x + 1, x))
        (1, Poly(2*x + 2, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'))

        >>> Poly(2*x**2 - 2, x).cancel(Poly(x**2 - 2*x + 1, x), include=True)
        (Poly(2*x + 2, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'))

        """
        dom, per, F, G = f._unify(g)
        if hasattr(F, 'cancel'):
            result = F.cancel(G, include=include)
        else:
            raise OperationNotSupported(f, 'cancel')
        if not include:
            if dom.has_assoc_Ring:
                dom = dom.get_ring()
            cp, cq, p, q = result
            cp = dom.to_sympy(cp)
            cq = dom.to_sympy(cq)
            return (cp / cq, per(p), per(q))
        else:
            return tuple(map(per, result))

    def make_monic_over_integers_by_scaling_roots(f):
        """
        Turn any univariate polynomial over :ref:`QQ` or :ref:`ZZ` into a monic
        polynomial over :ref:`ZZ`, by scaling the roots as necessary.

        Explanation
        ===========

        This operation can be performed whether or not *f* is irreducible; when
        it is, this can be understood as determining an algebraic integer
        generating the same field as a root of *f*.

        Examples
        ========

        >>> from sympy import Poly, S
        >>> from sympy.abc import x
        >>> f = Poly(x**2/2 + S(1)/4 * x + S(1)/8, x, domain='QQ')
        >>> f.make_monic_over_integers_by_scaling_roots()
        (Poly(x**2 + 2*x + 4, x, domain='ZZ'), 4)

        Returns
        =======

        Pair ``(g, c)``
            g is the polynomial

            c is the integer by which the roots had to be scaled

        """
        if not f.is_univariate or f.domain not in [ZZ, QQ]:
            raise ValueError('Polynomial must be univariate over ZZ or QQ.')
        if f.is_monic and f.domain == ZZ:
            return (f, ZZ.one)
        else:
            fm = f.monic()
            c, _ = fm.clear_denoms()
            return (fm.transform(Poly(fm.gen), c).to_ring(), c)

    def galois_group(f, by_name=False, max_tries=30, randomize=False):
        """
        Compute the Galois group of this polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x
        >>> f = Poly(x**4 - 2)
        >>> G, _ = f.galois_group(by_name=True)
        >>> print(G)
        S4TransitiveSubgroups.D4

        See Also
        ========

        sympy.polys.numberfields.galoisgroups.galois_group

        """
        from sympy.polys.numberfields.galoisgroups import _galois_group_degree_3, _galois_group_degree_4_lookup, _galois_group_degree_5_lookup_ext_factor, _galois_group_degree_6_lookup
        if not f.is_univariate or not f.is_irreducible or f.domain not in [ZZ, QQ]:
            raise ValueError('Polynomial must be irreducible and univariate over ZZ or QQ.')
        gg = {3: _galois_group_degree_3, 4: _galois_group_degree_4_lookup, 5: _galois_group_degree_5_lookup_ext_factor, 6: _galois_group_degree_6_lookup}
        max_supported = max(gg.keys())
        n = f.degree()
        if n > max_supported:
            raise ValueError(f'Only polynomials up to degree {max_supported} are supported.')
        elif n < 1:
            raise ValueError('Constant polynomial has no Galois group.')
        elif n == 1:
            from sympy.combinatorics.galois import S1TransitiveSubgroups
            name, alt = (S1TransitiveSubgroups.S1, True)
        elif n == 2:
            from sympy.combinatorics.galois import S2TransitiveSubgroups
            name, alt = (S2TransitiveSubgroups.S2, False)
        else:
            g, _ = f.make_monic_over_integers_by_scaling_roots()
            name, alt = gg[n](g, max_tries=max_tries, randomize=randomize)
        G = name if by_name else name.get_perm_group()
        return (G, alt)

    @property
    def is_zero(f):
        """
        Returns ``True`` if ``f`` is a zero polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(0, x).is_zero
        True
        >>> Poly(1, x).is_zero
        False

        """
        return f.rep.is_zero

    @property
    def is_one(f):
        """
        Returns ``True`` if ``f`` is a unit polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(0, x).is_one
        False
        >>> Poly(1, x).is_one
        True

        """
        return f.rep.is_one

    @property
    def is_sqf(f):
        """
        Returns ``True`` if ``f`` is a square-free polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 2*x + 1, x).is_sqf
        False
        >>> Poly(x**2 - 1, x).is_sqf
        True

        """
        return f.rep.is_sqf

    @property
    def is_monic(f):
        """
        Returns ``True`` if the leading coefficient of ``f`` is one.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x + 2, x).is_monic
        True
        >>> Poly(2*x + 2, x).is_monic
        False

        """
        return f.rep.is_monic

    @property
    def is_primitive(f):
        """
        Returns ``True`` if GCD of the coefficients of ``f`` is one.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(2*x**2 + 6*x + 12, x).is_primitive
        False
        >>> Poly(x**2 + 3*x + 6, x).is_primitive
        True

        """
        return f.rep.is_primitive

    @property
    def is_ground(f):
        """
        Returns ``True`` if ``f`` is an element of the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x, x).is_ground
        False
        >>> Poly(2, x).is_ground
        True
        >>> Poly(y, x).is_ground
        True

        """
        return f.rep.is_ground

    @property
    def is_linear(f):
        """
        Returns ``True`` if ``f`` is linear in all its variables.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x + y + 2, x, y).is_linear
        True
        >>> Poly(x*y + 2, x, y).is_linear
        False

        """
        return f.rep.is_linear

    @property
    def is_quadratic(f):
        """
        Returns ``True`` if ``f`` is quadratic in all its variables.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x*y + 2, x, y).is_quadratic
        True
        >>> Poly(x*y**2 + 2, x, y).is_quadratic
        False

        """
        return f.rep.is_quadratic

    @property
    def is_monomial(f):
        """
        Returns ``True`` if ``f`` is zero or has only one term.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(3*x**2, x).is_monomial
        True
        >>> Poly(3*x**2 + 1, x).is_monomial
        False

        """
        return f.rep.is_monomial

    @property
    def is_homogeneous(f):
        """
        Returns ``True`` if ``f`` is a homogeneous polynomial.

        A homogeneous polynomial is a polynomial whose all monomials with
        non-zero coefficients have the same total degree. If you want not
        only to check if a polynomial is homogeneous but also compute its
        homogeneous order, then use :func:`Poly.homogeneous_order`.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x*y, x, y).is_homogeneous
        True
        >>> Poly(x**3 + x*y, x, y).is_homogeneous
        False

        """
        return f.rep.is_homogeneous

    @property
    def is_irreducible(f):
        """
        Returns ``True`` if ``f`` has no factors over its domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 + x + 1, x, modulus=2).is_irreducible
        True
        >>> Poly(x**2 + 1, x, modulus=2).is_irreducible
        False

        """
        return f.rep.is_irreducible

    @property
    def is_univariate(f):
        """
        Returns ``True`` if ``f`` is a univariate polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x + 1, x).is_univariate
        True
        >>> Poly(x*y**2 + x*y + 1, x, y).is_univariate
        False
        >>> Poly(x*y**2 + x*y + 1, x).is_univariate
        True
        >>> Poly(x**2 + x + 1, x, y).is_univariate
        False

        """
        return len(f.gens) == 1

    @property
    def is_multivariate(f):
        """
        Returns ``True`` if ``f`` is a multivariate polynomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> Poly(x**2 + x + 1, x).is_multivariate
        False
        >>> Poly(x*y**2 + x*y + 1, x, y).is_multivariate
        True
        >>> Poly(x*y**2 + x*y + 1, x).is_multivariate
        False
        >>> Poly(x**2 + x + 1, x, y).is_multivariate
        True

        """
        return len(f.gens) != 1

    @property
    def is_cyclotomic(f):
        """
        Returns ``True`` if ``f`` is a cyclotomic polnomial.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1

        >>> Poly(f).is_cyclotomic
        False

        >>> g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1

        >>> Poly(g).is_cyclotomic
        True

        """
        return f.rep.is_cyclotomic

    def __abs__(f):
        return f.abs()

    def __neg__(f):
        return f.neg()

    @_polifyit
    def __add__(f, g):
        return f.add(g)

    @_polifyit
    def __radd__(f, g):
        return g.add(f)

    @_polifyit
    def __sub__(f, g):
        return f.sub(g)

    @_polifyit
    def __rsub__(f, g):
        return g.sub(f)

    @_polifyit
    def __mul__(f, g):
        return f.mul(g)

    @_polifyit
    def __rmul__(f, g):
        return g.mul(f)

    @_sympifyit('n', NotImplemented)
    def __pow__(f, n):
        if n.is_Integer and n >= 0:
            return f.pow(n)
        else:
            return NotImplemented

    @_polifyit
    def __divmod__(f, g):
        return f.div(g)

    @_polifyit
    def __rdivmod__(f, g):
        return g.div(f)

    @_polifyit
    def __mod__(f, g):
        return f.rem(g)

    @_polifyit
    def __rmod__(f, g):
        return g.rem(f)

    @_polifyit
    def __floordiv__(f, g):
        return f.quo(g)

    @_polifyit
    def __rfloordiv__(f, g):
        return g.quo(f)

    @_sympifyit('g', NotImplemented)
    def __truediv__(f, g):
        return f.as_expr() / g.as_expr()

    @_sympifyit('g', NotImplemented)
    def __rtruediv__(f, g):
        return g.as_expr() / f.as_expr()

    @_sympifyit('other', NotImplemented)
    def __eq__(self, other):
        f, g = (self, other)
        if not g.is_Poly:
            try:
                g = f.__class__(g, f.gens, domain=f.get_domain())
            except (PolynomialError, DomainError, CoercionFailed):
                return False
        if f.gens != g.gens:
            return False
        if f.rep.dom != g.rep.dom:
            return False
        return f.rep == g.rep

    @_sympifyit('g', NotImplemented)
    def __ne__(f, g):
        return not f == g

    def __bool__(f):
        return not f.is_zero

    def eq(f, g, strict=False):
        if not strict:
            return f == g
        else:
            return f._strict_eq(sympify(g))

    def ne(f, g, strict=False):
        return not f.eq(g, strict=strict)

    def _strict_eq(f, g):
        return isinstance(g, f.__class__) and f.gens == g.gens and f.rep.eq(g.rep, strict=True)
from sympy.core.numbers import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom
class ModuleElement(IntegerPowerable):
    """
    Represents an element of a :py:class:`~.Module`.

    NOTE: Should not be constructed directly. Use the
    :py:meth:`~.Module.__call__` method or the :py:func:`make_mod_elt()`
    factory function instead.
    """

    def __init__(self, module, col, denom=1):
        """
        Parameters
        ==========

        module : :py:class:`~.Module`
            The module to which this element belongs.
        col : :py:class:`~.DomainMatrix` over :ref:`ZZ`
            Column vector giving the numerators of the coefficients of this
            element.
        denom : int, optional (default=1)
            Denominator for the coefficients of this element.

        """
        self.module = module
        self.col = col
        self.denom = denom
        self._QQ_col = None

    def __repr__(self):
        r = str([int(c) for c in self.col.flat()])
        if self.denom > 1:
            r += f'/{self.denom}'
        return r

    def reduced(self):
        """
        Produce a reduced version of this ModuleElement, i.e. one in which the
        gcd of the denominator together with all numerator coefficients is 1.
        """
        if self.denom == 1:
            return self
        g = igcd(self.denom, *self.coeffs)
        if g == 1:
            return self
        return type(self)(self.module, (self.col / g).convert_to(ZZ), denom=self.denom // g)

    def reduced_mod_p(self, p):
        """
        Produce a version of this :py:class:`~.ModuleElement` in which all
        numerator coefficients have been reduced mod *p*.
        """
        return make_mod_elt(self.module, self.col.convert_to(FF(p)).convert_to(ZZ), denom=self.denom)

    @classmethod
    def from_int_list(cls, module, coeffs, denom=1):
        """
        Make a :py:class:`~.ModuleElement` from a list of ints (instead of a
        column vector).
        """
        col = to_col(coeffs)
        return cls(module, col, denom=denom)

    @property
    def n(self):
        """The length of this element's column."""
        return self.module.n

    def __len__(self):
        return self.n

    def column(self, domain=None):
        """
        Get a copy of this element's column, optionally converting to a domain.
        """
        return self.col.convert_to(domain)

    @property
    def coeffs(self):
        return self.col.flat()

    @property
    def QQ_col(self):
        """
        :py:class:`~.DomainMatrix` over :ref:`QQ`, equal to
        ``self.col / self.denom``, and guaranteed to be dense.

        See Also
        ========

        .Submodule.QQ_matrix

        """
        if self._QQ_col is None:
            self._QQ_col = (self.col / self.denom).to_dense()
        return self._QQ_col

    def to_parent(self):
        """
        Transform into a :py:class:`~.ModuleElement` belonging to the parent of
        this element's module.
        """
        if not isinstance(self.module, Submodule):
            raise ValueError('Not an element of a Submodule.')
        return make_mod_elt(self.module.parent, self.module.matrix * self.col, denom=self.module.denom * self.denom)

    def to_ancestor(self, anc):
        """
        Transform into a :py:class:`~.ModuleElement` belonging to a given
        ancestor of this element's module.

        Parameters
        ==========

        anc : :py:class:`~.Module`

        """
        if anc == self.module:
            return self
        else:
            return self.to_parent().to_ancestor(anc)

    def over_power_basis(self):
        """
        Transform into a :py:class:`~.PowerBasisElement` over our
        :py:class:`~.PowerBasis` ancestor.
        """
        e = self
        while not isinstance(e.module, PowerBasis):
            e = e.to_parent()
        return e

    def is_compat(self, other):
        """
        Test whether other is another :py:class:`~.ModuleElement` with same
        module.
        """
        return isinstance(other, ModuleElement) and other.module == self.module

    def unify(self, other):
        """
        Try to make a compatible pair of :py:class:`~.ModuleElement`, one
        equivalent to this one, and one equivalent to the other.

        Explanation
        ===========

        We search for the nearest common ancestor module for the pair of
        elements, and represent each one there.

        Returns
        =======

        Pair ``(e1, e2)``
            Each ``ei`` is a :py:class:`~.ModuleElement`, they belong to the
            same :py:class:`~.Module`, ``e1`` is equivalent to ``self``, and
            ``e2`` is equivalent to ``other``.

        Raises
        ======

        UnificationFailed
            If ``self`` and ``other`` have no common ancestor module.

        """
        if self.module == other.module:
            return (self, other)
        nca = self.module.nearest_common_ancestor(other.module)
        if nca is not None:
            return (self.to_ancestor(nca), other.to_ancestor(nca))
        raise UnificationFailed(f'Cannot unify {self} with {other}')

    def __eq__(self, other):
        if self.is_compat(other):
            return self.QQ_col == other.QQ_col
        return NotImplemented

    def equiv(self, other):
        """
        A :py:class:`~.ModuleElement` may test as equivalent to a rational
        number or another :py:class:`~.ModuleElement`, if they represent the
        same algebraic number.

        Explanation
        ===========

        This method is intended to check equivalence only in those cases in
        which it is easy to test; namely, when *other* is either a
        :py:class:`~.ModuleElement` that can be unified with this one (i.e. one
        which shares a common :py:class:`~.PowerBasis` ancestor), or else a
        rational number (which is easy because every :py:class:`~.PowerBasis`
        represents every rational number).

        Parameters
        ==========

        other : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.ModuleElement`

        Returns
        =======

        bool

        Raises
        ======

        UnificationFailed
            If ``self`` and ``other`` do not share a common
            :py:class:`~.PowerBasis` ancestor.

        """
        if self == other:
            return True
        elif isinstance(other, ModuleElement):
            a, b = self.unify(other)
            return a == b
        elif is_rat(other):
            if isinstance(self, PowerBasisElement):
                return self == self.module(0) * other
            else:
                return self.over_power_basis().equiv(other)
        return False

    def __add__(self, other):
        """
        A :py:class:`~.ModuleElement` can be added to a rational number, or to
        another :py:class:`~.ModuleElement`.

        Explanation
        ===========

        When the other summand is a rational number, it will be converted into
        a :py:class:`~.ModuleElement` (belonging to the first ancestor of this
        module that starts with unity).

        In all cases, the sum belongs to the nearest common ancestor (NCA) of
        the modules of the two summands. If the NCA does not exist, we return
        ``NotImplemented``.
        """
        if self.is_compat(other):
            d, e = (self.denom, other.denom)
            m = ilcm(d, e)
            u, v = (m // d, m // e)
            col = to_col([u * a + v * b for a, b in zip(self.coeffs, other.coeffs)])
            return type(self)(self.module, col, denom=m).reduced()
        elif isinstance(other, ModuleElement):
            try:
                a, b = self.unify(other)
            except UnificationFailed:
                return NotImplemented
            return a + b
        elif is_rat(other):
            return self + self.module.element_from_rational(other)
        return NotImplemented
    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        """
        A :py:class:`~.ModuleElement` can be multiplied by a rational number,
        or by another :py:class:`~.ModuleElement`.

        Explanation
        ===========

        When the multiplier is a rational number, the product is computed by
        operating directly on the coefficients of this
        :py:class:`~.ModuleElement`.

        When the multiplier is another :py:class:`~.ModuleElement`, the product
        will belong to the nearest common ancestor (NCA) of the modules of the
        two operands, and that NCA must have a multiplication table. If the NCA
        does not exist, we return ``NotImplemented``. If the NCA does not have
        a mult. table, ``ClosureFailure`` will be raised.
        """
        if self.is_compat(other):
            M = self.module.mult_tab()
            A, B = (self.col.flat(), other.col.flat())
            n = self.n
            C = [0] * n
            for u in range(n):
                for v in range(u, n):
                    c = A[u] * B[v]
                    if v > u:
                        c += A[v] * B[u]
                    if c != 0:
                        R = M[u][v]
                        for k in range(n):
                            C[k] += c * R[k]
            d = self.denom * other.denom
            return self.from_int_list(self.module, C, denom=d)
        elif isinstance(other, ModuleElement):
            try:
                a, b = self.unify(other)
            except UnificationFailed:
                return NotImplemented
            return a * b
        elif is_rat(other):
            a, b = get_num_denom(other)
            if a == b == 1:
                return self
            else:
                return make_mod_elt(self.module, self.col * a, denom=self.denom * b).reduced()
        return NotImplemented
    __rmul__ = __mul__

    def _zeroth_power(self):
        return self.module.one()

    def _first_power(self):
        return self

    def __floordiv__(self, a):
        if is_rat(a):
            a = QQ(a)
            return self * (1 / a)
        elif isinstance(a, ModuleElement):
            return self * (1 // a)
        return NotImplemented

    def __rfloordiv__(self, a):
        return a // self.over_power_basis()

    def __mod__(self, m):
        """
        Reduce this :py:class:`~.ModuleElement` mod a :py:class:`~.Submodule`.

        Parameters
        ==========

        m : int, :ref:`ZZ`, :ref:`QQ`, :py:class:`~.Submodule`
            If a :py:class:`~.Submodule`, reduce ``self`` relative to this.
            If an integer or rational, reduce relative to the
            :py:class:`~.Submodule` that is our own module times this constant.

        See Also
        ========

        .Submodule.reduce_element

        """
        if is_rat(m):
            m = m * self.module.whole_submodule()
        if isinstance(m, Submodule) and m.parent == self.module:
            return m.reduce_element(self)
        return NotImplemented
import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
class Dimensionality(dict):
    """
    """

    @property
    def ndims(self):
        return sum((abs(i) for i in self.simplified.values()))

    @property
    def simplified(self):
        if len(self):
            rq = 1 * unit_registry['dimensionless']
            for u, d in self.items():
                rq = rq * u.simplified ** d
            return rq.dimensionality
        else:
            return self

    @property
    def string(self):
        return markup.format_units(self)

    @property
    def unicode(self):
        return markup.format_units_unicode(self)

    @property
    def latex(self):
        return markup.format_units_latex(self)

    @property
    def html(self):
        return markup.format_units_html(self)

    def __hash__(self):
        res = hash(unit_registry['dimensionless'])
        for key in sorted(self.keys(), key=operator.attrgetter('format_order')):
            val = self[key]
            if val < 0:
                val -= 1
            res ^= hash((key, val))
        return res

    def __add__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError('can not add units of %s and %s' % (str(self), str(other)))
        return self.copy()
    __radd__ = __add__

    def __iadd__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError('can not add units of %s and %s' % (str(self), str(other)))
        return self

    def __sub__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError('can not subtract units of %s and %s' % (str(self), str(other)))
        return self.copy()
    __rsub__ = __sub__

    def __isub__(self, other):
        assert_isinstance(other, Dimensionality)
        try:
            assert self == other
        except AssertionError:
            raise ValueError('can not add units of %s and %s' % (str(self), str(other)))
        return self

    def __mul__(self, other):
        assert_isinstance(other, Dimensionality)
        new = Dimensionality(self)
        for unit, power in other.items():
            try:
                new[unit] += power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = power
        return new

    def __imul__(self, other):
        assert_isinstance(other, Dimensionality)
        for unit, power in other.items():
            try:
                self[unit] += power
                if self[unit] == 0:
                    self.pop(unit)
            except KeyError:
                self[unit] = power
        return self

    def __truediv__(self, other):
        assert_isinstance(other, Dimensionality)
        new = Dimensionality(self)
        for unit, power in other.items():
            try:
                new[unit] -= power
                if new[unit] == 0:
                    new.pop(unit)
            except KeyError:
                new[unit] = -power
        return new

    def __itruediv__(self, other):
        assert_isinstance(other, Dimensionality)
        for unit, power in other.items():
            try:
                self[unit] -= power
                if self[unit] == 0:
                    self.pop(unit)
            except KeyError:
                self[unit] = -power
        return self

    def __pow__(self, other):
        try:
            assert np.isscalar(other)
        except AssertionError:
            raise TypeError('exponent must be a scalar, got %r' % other)
        if other == 0:
            return Dimensionality()
        new = Dimensionality(self)
        for i in new:
            new[i] *= other
        return new

    def __ipow__(self, other):
        try:
            assert np.isscalar(other)
        except AssertionError:
            raise TypeError('exponent must be a scalar, got %r' % other)
        if other == 0:
            self.clear()
            return self
        for i in self:
            self[i] *= other
        return self

    def __repr__(self):
        return 'Dimensionality({%s})' % ', '.join(['%s: %s' % (u.name, e) for u, e in self.items()])

    def __str__(self):
        if markup.config.use_unicode:
            return self.unicode
        else:
            return self.string

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return hash(self) != hash(other)
    __neq__ = __ne__

    def __gt__(self, other):
        return self.ndims > other.ndims

    def __ge__(self, other):
        return self.ndims >= other.ndims

    def __lt__(self, other):
        return self.ndims < other.ndims

    def __le__(self, other):
        return self.ndims <= other.ndims

    def copy(self):
        return Dimensionality(self)
from __future__ import unicode_literals
class ValueMixin(object):
    """Provides simplistic but often sufficient comparison and string methods."""

    def __eq__(self, other):
        return getattr(other, '__dict__', None) == self.__dict__

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(frozenset(list(self.__dict__.items())))

    def __repr__(self):
        """Returns a string representation like `MyClass(foo=23, bar=skidoo)`."""
        d = self.__dict__
        attrs = ['{}={}'.format(key, d[key]) for key in sorted(d)]
        return '{}({})'.format(self.__class__.__name__, ', '.join(attrs))
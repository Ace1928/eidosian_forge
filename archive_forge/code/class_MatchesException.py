import sys
from ._basic import MatchesRegex
from ._higherorder import AfterPreproccessing
from ._impl import (
class MatchesException(Matcher):
    """Match an exc_info tuple against an exception instance or type."""

    def __init__(self, exception, value_re=None):
        """Create a MatchesException that will match exc_info's for exception.

        :param exception: Either an exception instance or type.
            If an instance is given, the type and arguments of the exception
            are checked. If a type is given only the type of the exception is
            checked. If a tuple is given, then as with isinstance, any of the
            types in the tuple matching is sufficient to match.
        :param value_re: If 'exception' is a type, and the matchee exception
            is of the right type, then match against this.  If value_re is a
            string, then assume value_re is a regular expression and match
            the str() of the exception against it.  Otherwise, assume value_re
            is a matcher, and match the exception against it.
        """
        Matcher.__init__(self)
        self.expected = exception
        if isinstance(value_re, str):
            value_re = AfterPreproccessing(str, MatchesRegex(value_re), False)
        self.value_re = value_re
        expected_type = type(self.expected)
        self._is_instance = not any((issubclass(expected_type, class_type) for class_type in (type, tuple)))

    def match(self, other):
        if type(other) != tuple:
            return Mismatch('%r is not an exc_info tuple' % other)
        expected_class = self.expected
        if self._is_instance:
            expected_class = expected_class.__class__
        if not issubclass(other[0], expected_class):
            return Mismatch(f'{other[0]!r} is not a {expected_class!r}')
        if self._is_instance:
            if other[1].args != self.expected.args:
                return Mismatch('{} has different arguments to {}.'.format(_error_repr(other[1]), _error_repr(self.expected)))
        elif self.value_re is not None:
            return self.value_re.match(other[1])

    def __str__(self):
        if self._is_instance:
            return 'MatchesException(%s)' % _error_repr(self.expected)
        return 'MatchesException(%s)' % repr(self.expected)
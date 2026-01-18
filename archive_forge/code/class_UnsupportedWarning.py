import collections
import functools
import textwrap
import warnings
from packaging import version
from datetime import date
class UnsupportedWarning(DeprecatedWarning):
    """A warning class for methods to be removed

    This is a subclass of :class:`~deprecation.DeprecatedWarning` and is used
    to output a proper message about a function being unsupported.
    Additionally, the :func:`~deprecation.fail_if_not_removed` decorator
    will handle this warning and cause any tests to fail if the system
    under test uses code that raises this warning.
    """

    def __str__(self):
        parts = collections.defaultdict(str)
        parts['function'] = self.function
        parts['removed'] = self.removed_in
        if self.details:
            parts['details'] = ' %s' % self.details
        return '%(function)s is unsupported as of %(removed)s.%(details)s' % parts
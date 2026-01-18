import collections
import functools
import textwrap
import warnings
from packaging import version
from datetime import date
class DeprecatedWarning(DeprecationWarning):
    """A warning class for deprecated methods

    This is a specialization of the built-in :class:`DeprecationWarning`,
    adding parameters that allow us to get information into the __str__
    that ends up being sent through the :mod:`warnings` system.
    The attributes aren't able to be retrieved after the warning gets
    raised and passed through the system as only the class--not the
    instance--and message are what gets preserved.

    :param function: The function being deprecated.
    :param deprecated_in: The version that ``function`` is deprecated in
    :param removed_in: The version or :class:`datetime.date` specifying
                       when ``function`` gets removed.
    :param details: Optional details about the deprecation. Most often
                    this will include directions on what to use instead
                    of the now deprecated code.
    """

    def __init__(self, function, deprecated_in, removed_in, details=''):
        self.function = function
        self.deprecated_in = deprecated_in
        self.removed_in = removed_in
        self.details = details
        super(DeprecatedWarning, self).__init__(function, deprecated_in, removed_in, details)

    def __str__(self):
        parts = collections.defaultdict(str)
        parts['function'] = self.function
        if self.deprecated_in:
            parts['deprecated'] = ' as of %s' % self.deprecated_in
        if self.removed_in:
            parts['removed'] = ' and will be removed {} {}'.format('on' if isinstance(self.removed_in, date) else 'in', self.removed_in)
        if any([self.deprecated_in, self.removed_in, self.details]):
            parts['period'] = '.'
        if self.details:
            parts['details'] = ' %s' % self.details
        return '%(function)s is deprecated%(deprecated)s%(removed)s%(period)s%(details)s' % parts
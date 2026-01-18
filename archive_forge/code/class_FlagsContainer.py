from pprint import pformat
from .py3compat import MutableMapping
class FlagsContainer(Container):
    """
    A container providing pretty-printing for flags.

    Only set flags are displayed.
    """

    @recursion_lock('<...>')
    def __str__(self):
        d = dict(((k, self[k]) for k in self if self[k] and (not k.startswith('_'))))
        return '%s(%s)' % (self.__class__.__name__, pformat(d))
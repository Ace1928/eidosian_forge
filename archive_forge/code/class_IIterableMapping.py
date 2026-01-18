from zope.interface import Interface
from zope.interface.common import collections
class IIterableMapping(IEnumerableMapping):
    """A mapping that has distinct methods for iterating
    without copying.

    """
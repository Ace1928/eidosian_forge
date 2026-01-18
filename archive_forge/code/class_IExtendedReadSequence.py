from zope.interface import Interface
from zope.interface.common import collections
class IExtendedReadSequence(IReadSequence):
    """Full read interface for lists"""

    def count(item):
        """Return number of occurrences of value"""

    def index(item, *args):
        """index(value, [start, [stop]]) -> int

        Return first index of *value*
        """
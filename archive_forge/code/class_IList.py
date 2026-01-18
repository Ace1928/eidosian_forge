from zope.interface import classImplements
from zope.interface.common import collections
from zope.interface.common import io
from zope.interface.common import numbers
class IList(collections.IMutableSequence):
    """
    Interface for :class:`list`
    """
    extra_classes = (list,)

    def sort(key=None, reverse=False):
        """
        Sort the list in place and return None.

        *key* and *reverse* must be passed by name only.
        """
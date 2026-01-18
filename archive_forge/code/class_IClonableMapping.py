from zope.interface import Interface
from zope.interface.common import collections
class IClonableMapping(Interface):
    """Something that can produce a copy of itself.

    This is available in `dict`.
    """

    def copy():
        """return copy of dict"""
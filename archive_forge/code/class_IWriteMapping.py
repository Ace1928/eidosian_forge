from zope.interface import Interface
from zope.interface.common import collections
class IWriteMapping(Interface):
    """Mapping methods for changing data"""

    def __delitem__(key):
        """Delete a value from the mapping using the key."""

    def __setitem__(key, value):
        """Set a new item in the mapping."""
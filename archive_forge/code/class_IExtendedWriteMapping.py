from zope.interface import Interface
from zope.interface.common import collections
class IExtendedWriteMapping(IWriteMapping):
    """Additional mutation methods.

    These are all provided by `dict`.
    """

    def clear():
        """delete all items"""

    def update(d):
        """ Update D from E: for k in E.keys(): D[k] = E[k]"""

    def setdefault(key, default=None):
        """D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D"""

    def pop(k, default=None):
        """
        pop(k[,default]) -> value

        Remove specified key and return the corresponding value.

        If key is not found, *default* is returned if given, otherwise
        `KeyError` is raised. Note that *default* must not be passed by
        name.
        """

    def popitem():
        """remove and return some (key, value) pair as a
        2-tuple; but raise KeyError if mapping is empty"""
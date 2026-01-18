from zope.interface import Interface
from zope.interface.common import collections
class IFullMapping(collections.IMutableMapping, IExtendedReadMapping, IExtendedWriteMapping, IClonableMapping, IMapping):
    """
    Full mapping interface.

    Most uses of this interface should instead use
    :class:`~zope.interface.commons.collections.IMutableMapping` (one of the
    bases of this interface). The required methods are the same.

    .. versionchanged:: 5.0.0
       Extend ``IMutableMapping``
    """
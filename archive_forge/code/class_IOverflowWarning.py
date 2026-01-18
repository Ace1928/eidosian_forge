from zope.interface import Interface
from zope.interface import classImplements
class IOverflowWarning(IWarning):
    """Deprecated, no standard class implements this.

    This was the interface for ``OverflowWarning`` prior to Python 2.5,
    but that class was removed for all versions after that.
    """
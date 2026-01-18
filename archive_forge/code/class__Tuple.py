from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from twisted.internet.defer import Deferred
class _Tuple(_Container):
    """
    Manage tuple containing circular references. Deprecated: use C{_Container}
    instead.
    """

    def __init__(self, l):
        """
        @param l: The list of object which may contain some not yet referenced
        objects.
        """
        _Container.__init__(self, l, tuple)
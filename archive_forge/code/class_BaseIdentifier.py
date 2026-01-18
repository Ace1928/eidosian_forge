from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
class BaseIdentifier(object):
    """Base class for all IEEE identifiers."""
    __slots__ = ('_value', '__weakref__')

    def __init__(self):
        self._value = None

    def __int__(self):
        """:return: integer value of this identifier"""
        return self._value

    def __index__(self):
        """
        :return: return the integer value of this identifier.
        """
        return self._value
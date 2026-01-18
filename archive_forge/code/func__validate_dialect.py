from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
def _validate_dialect(self, value):
    if value is None:
        if self._module is _eui64:
            return eui64_base
        else:
            return mac_eui48
    elif hasattr(value, 'word_size') and hasattr(value, 'word_fmt'):
        return value
    else:
        raise TypeError('custom dialects should subclass mac_eui48!')
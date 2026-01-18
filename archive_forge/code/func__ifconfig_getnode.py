import os
import sys
from enum import Enum, _simple_enum
def _ifconfig_getnode():
    """Get the hardware address on Unix by running ifconfig."""
    keywords = (b'hwaddr', b'ether', b'address:', b'lladdr')
    for args in ('', '-a', '-av'):
        mac = _find_mac_near_keyword('ifconfig', args, keywords, lambda i: i + 1)
        if mac:
            return mac
    return None
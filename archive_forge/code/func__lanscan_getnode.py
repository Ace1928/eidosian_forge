import os
import sys
from enum import Enum, _simple_enum
def _lanscan_getnode():
    """Get the hardware address on Unix by running lanscan."""
    return _find_mac_near_keyword('lanscan', '-ai', [b'lan0'], lambda i: 0)
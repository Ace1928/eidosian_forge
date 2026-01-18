import os
import sys
from enum import Enum, _simple_enum
def _ipconfig_getnode():
    """[DEPRECATED] Get the hardware address on Windows."""
    return _windll_getnode()
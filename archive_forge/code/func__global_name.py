from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
@staticmethod
def _global_name(name):
    if name.startswith('PROTOCOL_ERROR_'):
        return name
    else:
        return 'EVENT_' + name
import os
import sys
from enum import Enum, _simple_enum
def _is_universal(mac):
    return not mac & 1 << 41
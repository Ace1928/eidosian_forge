import enum
import os
import struct
from typing import IO, Optional, Tuple
class EIClass(enum.IntEnum):
    C32 = 1
    C64 = 2
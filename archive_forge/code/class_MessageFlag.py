from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
class MessageFlag(IntFlag):
    no_reply_expected = 1
    no_auto_start = 2
    allow_interactive_authorization = 4
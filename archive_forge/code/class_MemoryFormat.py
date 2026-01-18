from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
class MemoryFormat(IntEnum):
    Unknown = 0
    ContiguousFormat = 1
    ChannelsLast = 2
    ChannelsLast3d = 3
    PreserveFormat = 4
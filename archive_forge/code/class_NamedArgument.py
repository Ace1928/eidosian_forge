from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass
class NamedArgument:
    name: str
    arg: Argument
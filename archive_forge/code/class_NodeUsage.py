import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class NodeUsage:
    usage: List[ResourceUsage]
    idle_time_ms: int
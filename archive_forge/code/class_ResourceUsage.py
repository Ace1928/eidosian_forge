import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class ResourceUsage:
    resource_name: str = ''
    total: float = 0.0
    used: float = 0.0
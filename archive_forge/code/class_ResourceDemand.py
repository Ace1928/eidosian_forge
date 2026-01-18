import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class ResourceDemand:
    bundles_by_count: List[ResourceRequestByCount]
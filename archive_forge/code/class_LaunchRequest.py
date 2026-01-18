import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class LaunchRequest:

    class Status(Enum):
        FAILED = 'FAILED'
        PENDING = 'PENDING'
    instance_type_name: str
    ray_node_type_name: str
    count: int
    state: Status
    request_ts_s: int
    failed_ts_s: Optional[int] = None
    details: Optional[str] = None
from enum import Enum
from dataclasses import dataclass
from datetime import timedelta
@dataclass
class BroadcastOptions:
    root_rank = 0
    root_tensor = 0
    timeout_ms = unset_timeout_ms
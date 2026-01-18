from enum import Enum
from dataclasses import dataclass
from datetime import timedelta
@dataclass
class BarrierOptions:
    timeout_ms = unset_timeout_ms
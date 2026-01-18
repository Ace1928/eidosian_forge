import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional
from ray.serve._private.constants import SERVE_LOGGER_NAME
@dataclass(order=True)
class TimeStampedValue:
    timestamp: float
    value: float = field(compare=False)
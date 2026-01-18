import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from ray.autoscaler._private.constants import (
from ray.autoscaler.node_launch_exception import NodeLaunchException
@dataclass
class NodeAvailabilityRecord:
    node_type: str
    is_available: bool
    last_checked_timestamp: float
    unavailable_node_information: Optional[UnavailableNodeInformation]
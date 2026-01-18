import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from ray.autoscaler._private.constants import (
from ray.autoscaler.node_launch_exception import NodeLaunchException
@dataclass
class UnavailableNodeInformation:
    category: str
    description: str
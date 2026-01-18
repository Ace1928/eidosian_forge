import copy
import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ray.autoscaler.v2.instance_manager.storage import Storage, StoreStatus
from ray.core.generated.instance_manager_pb2 import Instance
@dataclass
class InstanceUpdateEvent:
    """Notifies the status change of an instance."""
    instance_id: str
    new_status: int
    new_ray_status: int = Instance.RAY_STATUS_UNKOWN
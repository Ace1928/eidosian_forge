from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
class _Tag(Enum):
    """
    A set of special values used for the method arguments de/construction.

    See ``DeferredExecution._deconstruct()`` for details.
    """
    CHAIN = 0
    REF = 1
    LIST = 2
    END = 3
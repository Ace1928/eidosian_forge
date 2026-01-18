from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger

        Use a single instance on deserialization.

        Returns
        -------
        str
            Returns the ``_REMOTE_EXEC`` attribute name.
        
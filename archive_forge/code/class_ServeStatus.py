import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
@dataclass(eq=True)
class ServeStatus:
    """Describes the status of Serve.

    Attributes:
        proxies: The proxy actors running on each node in the cluster.
            A map from node ID to proxy status.
        applications: The live applications in the cluster.
        target_capacity: the target capacity percentage for all replicas across the
            cluster.
    """
    proxies: Dict[str, ProxyStatus] = field(default_factory=dict)
    applications: Dict[str, ApplicationStatusOverview] = field(default_factory=dict)
    target_capacity: Optional[float] = TARGET_CAPACITY_FIELD
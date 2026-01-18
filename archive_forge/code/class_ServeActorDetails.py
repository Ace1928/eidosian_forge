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
@PublicAPI(stability='stable')
class ServeActorDetails(BaseModel, frozen=True):
    node_id: Optional[str] = Field(description='ID of the node that the actor is running on.')
    node_ip: Optional[str] = Field(description='IP address of the node that the actor is running on.')
    actor_id: Optional[str] = Field(description='Actor ID.')
    actor_name: Optional[str] = Field(description='Actor name.')
    worker_id: Optional[str] = Field(description='Worker ID.')
    log_file_path: Optional[str] = Field(description="The relative path to the Serve actor's log file from the ray logs directory.")
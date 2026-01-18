import json
import logging
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional
from zlib import crc32
from ray._private.pydantic_compat import BaseModel
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.utils import DeploymentOptionUpdateType, get_random_letters
from ray.serve.generated.serve_pb2 import DeploymentVersion as DeploymentVersionProto
class VersionedReplica(ABC):

    @property
    def version(self) -> DeploymentVersion:
        pass

    def update_state(self, state):
        pass
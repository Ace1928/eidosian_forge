import base64
import json
from ray import cloudpickle
from enum import Enum, unique
import hashlib
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import ray
from ray import ObjectRef
from ray._private.utils import get_or_create_event_loop
from ray.util.annotations import PublicAPI
@ray.remote
def _hash(obj: Any) -> bytes:
    m = hashlib.sha256()
    m.update(cloudpickle.dumps(obj))
    return m.digest()
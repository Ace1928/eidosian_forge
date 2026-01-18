import weakref
from dataclasses import dataclass
import logging
from typing import List, TypeVar, Optional, Dict, Type, Tuple
import ray
from ray.actor import ActorHandle
from ray.util.annotations import Deprecated
from ray._private.utils import get_ray_doc_version
@dataclass
class ActorConfig:
    num_cpus: float
    num_gpus: float
    resources: Optional[Dict[str, float]]
    init_args: Tuple
    init_kwargs: Dict
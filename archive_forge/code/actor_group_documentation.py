import weakref
from dataclasses import dataclass
import logging
from typing import List, TypeVar, Optional, Dict, Type, Tuple
import ray
from ray.actor import ActorHandle
from ray.util.annotations import Deprecated
from ray._private.utils import get_ray_doc_version
Adds ``num_actors`` to this ActorGroup.

        Args:
            num_actors: The number of actors to add.
        
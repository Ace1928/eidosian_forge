import contextlib
from dataclasses import dataclass
import logging
import os
import ray
from ray import cloudpickle
from ray.types import ObjectRef
from ray.workflow import common, workflow_storage
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING
from collections import ChainMap
import io
def _obj_id_to_key(object_id: str) -> str:
    return os.path.join(OBJECTS_DIR, object_id)
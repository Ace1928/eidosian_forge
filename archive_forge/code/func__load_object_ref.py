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
def _load_object_ref(key: str, workflow_id: str) -> ray.ObjectRef:
    global _object_cache
    if _object_cache is None:
        return _load_ref_helper.remote(key, workflow_id)
    if _object_cache is None:
        return _load_ref_helper.remote(key, workflow_id)
    if key not in _object_cache:
        _object_cache[key] = _load_ref_helper.remote(key, workflow_id)
    return _object_cache[key]
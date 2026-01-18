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
@ray.remote
def _load_ref_helper(key: str, workflow_id: str):
    storage = workflow_storage.WorkflowStorage(workflow_id)
    return storage._get(key)
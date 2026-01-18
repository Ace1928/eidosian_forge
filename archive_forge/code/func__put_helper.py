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
@ray.remote(num_cpus=0)
def _put_helper(identifier: str, obj: Any, workflow_id: str) -> None:
    if isinstance(obj, ray.ObjectRef):
        raise NotImplementedError('Workflow does not support checkpointing nested object references yet.')
    key = _obj_id_to_key(identifier)
    dump_to_storage(key, obj, workflow_id, workflow_storage.WorkflowStorage(workflow_id), update_existing=False)
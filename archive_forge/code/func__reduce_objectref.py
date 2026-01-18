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
def _reduce_objectref(workflow_id: str, obj_ref: ObjectRef, tasks: List[ObjectRef]):
    manager = get_or_create_manager()
    key, task = ray.get(manager.save_objectref.remote((obj_ref,), workflow_id))
    assert task
    tasks.append(task)
    return (_load_object_ref, (key, workflow_id))
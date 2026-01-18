import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import cloudpickle
from ray._private import storage
from ray.types import ObjectRef
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowNotFoundError
from ray.workflow import workflow_context
from ray.workflow import serialization
from ray.workflow import serialization_context
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.storage import DataLoadError, DataSaveError, KeyNotFoundError
def _load_workflow_metadata():
    if not self._scan('', ignore_errors=True):
        raise ValueError("No such workflow_id '{}'".format(self._workflow_id))
    tasks = [self._get(self._key_workflow_metadata(), True, True), self._get(self._key_workflow_user_metadata(), True, True), self._get(self._key_workflow_prerun_metadata(), True, True), self._get(self._key_workflow_postrun_metadata(), True, True)]
    (status_metadata, _), (user_metadata, _), (prerun_metadata, _), (postrun_metadata, _) = tasks
    status_metadata = status_metadata or {}
    user_metadata = user_metadata or {}
    prerun_metadata = prerun_metadata or {}
    postrun_metadata = postrun_metadata or {}
    metadata = status_metadata
    metadata['user_metadata'] = user_metadata
    metadata['stats'] = {**prerun_metadata, **postrun_metadata}
    return metadata
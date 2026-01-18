import functools
import logging
from typing import Dict, Set, List, Tuple, Union, Optional, Any
import time
import uuid
import ray
from ray.dag import DAGNode
from ray.dag.input_node import DAGInputData
from ray.remote_function import RemoteFunction
from ray.workflow.common import (
from ray.workflow import serialization, workflow_access, workflow_context
from ray.workflow.event_listener import EventListener, EventListenerType, TimerListener
from ray.workflow.workflow_storage import WorkflowStorage
from ray.workflow.workflow_state_from_dag import workflow_state_from_dag
from ray.util.annotations import PublicAPI
from ray._private.usage import usage_lib
@client_mode_wrap
def _try_checkpoint_workflow(workflow_state) -> bool:
    ws = WorkflowStorage(workflow_id)
    ws.save_workflow_user_metadata(metadata)
    try:
        ws.get_entrypoint_task_id()
        return True
    except Exception:
        ws.save_workflow_execution_state('', workflow_state)
        return False
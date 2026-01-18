import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef
def _keep_workflow_refs(index: int):
    return _KeepWorkflowRefs(index)
import contextlib
from typing import List, Any, Dict
from ray.util.serialization import register_serializer, deregister_serializer
from ray.workflow.common import WorkflowRef
class _KeepWorkflowRefs:

    def __init__(self, index: int):
        self._index = index

    def __reduce__(self):
        return (_resolve_workflow_refs, (self._index,))
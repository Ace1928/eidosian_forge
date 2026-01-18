from typing import Any, Dict, List, Union, Optional
from ray.dag import DAGNode
from ray.dag.format_utils import get_dag_node_str
from ray.experimental.gradio_utils import type_to_string
from ray.util.annotations import DeveloperAPI
def _in_context_manager(self) -> bool:
    """Return if InputNode is created in context manager."""
    if not self._bound_other_args_to_resolve or IN_CONTEXT_MANAGER not in self._bound_other_args_to_resolve:
        return False
    else:
        return self._bound_other_args_to_resolve[IN_CONTEXT_MANAGER]
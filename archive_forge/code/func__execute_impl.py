from typing import Any, Dict, List
from ray.dag import DAGNode
from ray.dag.constants import DAGNODE_TYPE_KEY
from ray.dag.format_utils import get_dag_node_str
from ray.serve.handle import RayServeHandle
def _execute_impl(self, *args, **kwargs) -> RayServeHandle:
    """Does not call into anything or produce a new value, as the time
        this function gets called, all child nodes are already resolved to
        ObjectRefs.
        """
    return self._deployment_handle.options(use_new_handle_api=False)
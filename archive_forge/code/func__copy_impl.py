from typing import Any, Dict, List
from ray.dag import DAGNode
from ray.dag.constants import DAGNODE_TYPE_KEY
from ray.dag.format_utils import get_dag_node_str
from ray.serve.handle import RayServeHandle
def _copy_impl(self, new_args: List[Any], new_kwargs: Dict[str, Any], new_options: Dict[str, Any], new_other_args_to_resolve: Dict[str, Any]) -> 'DeploymentExecutorNode':
    return DeploymentExecutorNode(self._deployment_handle, new_args, new_kwargs)
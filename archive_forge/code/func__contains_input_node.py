import ray
from ray.dag.dag_node import DAGNode
from ray.dag.input_node import InputNode
from ray.dag.format_utils import get_dag_node_str
from ray.dag.constants import (
from ray.util.annotations import DeveloperAPI
from typing import Any, Dict, List, Optional, Tuple
def _contains_input_node(self) -> bool:
    """Check if InputNode is used in children DAGNodes with current node
        as the root.
        """
    children_dag_nodes = self._get_all_child_nodes()
    for child in children_dag_nodes:
        if isinstance(child, InputNode):
            return True
    return False
from ray.dag import DAGNode
import os
import tempfile
from ray.dag.utils import _DAGNodeNameGenerator
from ray.util.annotations import DeveloperAPI
def _get_nodes_and_edges(dag: DAGNode):
    """Get all unique nodes and edges in the DAG.

    A basic dfs with memorization to get all unique nodes
    and edges in the DAG.
    Unique nodes will be used to generate unique names,
    while edges will be used to construct the graph.
    """
    edges = []
    nodes = []

    def _dfs(node):
        nodes.append(node)
        for child_node in node._get_all_child_nodes():
            edges.append((child_node, node))
        return node
    dag.apply_recursive(_dfs)
    return (nodes, edges)
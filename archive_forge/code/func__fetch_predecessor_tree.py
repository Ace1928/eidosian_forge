import functools
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import tree
from taskflow.utils import misc
def _fetch_predecessor_tree(graph, atom):
    """Creates a tree of predecessors, rooted at given atom."""
    root = tree.Node(atom)
    stack = [(root, atom)]
    while stack:
        parent, node = stack.pop()
        for pred_node in graph.predecessors(node):
            pred_node_data = graph.nodes[pred_node]
            if pred_node_data['kind'] == compiler.FLOW_END:
                for pred_pred_node in graph.predecessors(pred_node):
                    stack.append((parent, pred_pred_node))
            else:
                child = tree.Node(pred_node, **pred_node_data)
                parent.add(child)
                stack.append((child, pred_node))
    return root
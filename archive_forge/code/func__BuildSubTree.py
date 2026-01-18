from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _BuildSubTree(plan_nodes, node):
    """Helper for building the subtree of a query plan node.

  Args:
    plan_nodes (spanner_v1_messages.PlanNode[]): The plan_nodes from the server
      response. Plan nodes are topologically sorted.
    node (spanner_v1_messages.PlanNode): The root node of the subtree to be
      built.

  Returns:
    A Node object.
  """
    children = None
    if node.childLinks:
        children = [_BuildSubTree(plan_nodes, plan_nodes[link.childIndex]) for link in node.childLinks]
    return Node(node, children)
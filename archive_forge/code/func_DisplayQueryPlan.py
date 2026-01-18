from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def DisplayQueryPlan(result, out):
    """Displays a graphical query plan for a query.

  Args:
    result (spanner_v1_messages.ResultSet): The server response to a query.
    out: Output stream to which we print.
  """
    node_tree_root = _ConvertToTree(result.stats.queryPlan.planNodes)
    node_tree_root.PrettyPrint(out)
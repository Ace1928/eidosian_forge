from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkPlanGraphNodeWrapper(_messages.Message):
    """Wrapper user to represent either a node or a cluster.

  Fields:
    cluster: A SparkPlanGraphCluster attribute.
    node: A SparkPlanGraphNode attribute.
  """
    cluster = _messages.MessageField('SparkPlanGraphCluster', 1)
    node = _messages.MessageField('SparkPlanGraphNode', 2)
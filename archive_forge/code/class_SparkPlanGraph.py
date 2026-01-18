from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkPlanGraph(_messages.Message):
    """A graph used for storing information of an executionPlan of DataFrame.

  Fields:
    edges: A SparkPlanGraphEdge attribute.
    executionId: A string attribute.
    nodes: A SparkPlanGraphNodeWrapper attribute.
  """
    edges = _messages.MessageField('SparkPlanGraphEdge', 1, repeated=True)
    executionId = _messages.IntegerField(2)
    nodes = _messages.MessageField('SparkPlanGraphNodeWrapper', 3, repeated=True)
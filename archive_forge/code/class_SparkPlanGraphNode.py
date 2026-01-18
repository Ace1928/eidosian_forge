from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkPlanGraphNode(_messages.Message):
    """Represents a node in the spark plan tree.

  Fields:
    desc: A string attribute.
    metrics: A SqlPlanMetric attribute.
    name: A string attribute.
    sparkPlanGraphNodeId: A string attribute.
  """
    desc = _messages.StringField(1)
    metrics = _messages.MessageField('SqlPlanMetric', 2, repeated=True)
    name = _messages.StringField(3)
    sparkPlanGraphNodeId = _messages.IntegerField(4)
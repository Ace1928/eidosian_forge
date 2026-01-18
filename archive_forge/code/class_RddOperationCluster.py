from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RddOperationCluster(_messages.Message):
    """A grouping of nodes representing higher level constructs (stage, job
  etc.).

  Fields:
    childClusters: A RddOperationCluster attribute.
    childNodes: A RddOperationNode attribute.
    name: A string attribute.
    rddClusterId: A string attribute.
  """
    childClusters = _messages.MessageField('RddOperationCluster', 1, repeated=True)
    childNodes = _messages.MessageField('RddOperationNode', 2, repeated=True)
    name = _messages.StringField(3)
    rddClusterId = _messages.StringField(4)
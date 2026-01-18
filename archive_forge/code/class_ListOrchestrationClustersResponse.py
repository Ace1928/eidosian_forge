from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOrchestrationClustersResponse(_messages.Message):
    """Message for response to listing OrchestrationClusters.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    orchestrationClusters: The list of OrchestrationCluster
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    orchestrationClusters = _messages.MessageField('OrchestrationCluster', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)
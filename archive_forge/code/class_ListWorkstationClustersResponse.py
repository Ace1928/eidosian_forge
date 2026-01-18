from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkstationClustersResponse(_messages.Message):
    """Response message for ListWorkstationClusters.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    unreachable: Unreachable resources.
    workstationClusters: The requested workstation clusters.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    workstationClusters = _messages.MessageField('WorkstationCluster', 3, repeated=True)
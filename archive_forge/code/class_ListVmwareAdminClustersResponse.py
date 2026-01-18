from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVmwareAdminClustersResponse(_messages.Message):
    """Response message for listing VMware admin clusters.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return. If the token is not empty this means that more results are
      available and should be retrieved by repeating the request with the
      provided page token.
    unreachable: Locations that could not be reached.
    vmwareAdminClusters: The list of VMware admin cluster.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    vmwareAdminClusters = _messages.MessageField('VmwareAdminCluster', 3, repeated=True)
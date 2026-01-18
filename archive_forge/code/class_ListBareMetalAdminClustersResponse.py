from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBareMetalAdminClustersResponse(_messages.Message):
    """Response message for listing bare metal admin clusters.

  Fields:
    bareMetalAdminClusters: The list of bare metal admin cluster.
    nextPageToken: A token identifying a page of results the server should
      return. If the token is not empty this means that more results are
      available and should be retrieved by repeating the request with the
      provided page token.
    unreachable: Locations that could not be reached.
  """
    bareMetalAdminClusters = _messages.MessageField('BareMetalAdminCluster', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)
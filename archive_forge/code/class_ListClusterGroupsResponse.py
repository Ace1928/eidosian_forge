from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListClusterGroupsResponse(_messages.Message):
    """A ListClusterGroupsResponse object.

  Fields:
    clusterGroups: A list of cluster groups.
    nextPageToken: A token, which can be send as `page_token` to retrieve the
      next page. If you omit this field, there are no subsequent pages.
    unreachable: List of locations that could not be reached.
  """
    clusterGroups = _messages.MessageField('ClusterGroup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)
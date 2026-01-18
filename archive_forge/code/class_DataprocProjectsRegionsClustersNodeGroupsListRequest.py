from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersNodeGroupsListRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersNodeGroupsListRequest object.

  Fields:
    pageSize: The maximum number of node groups to return. The service may
      return fewer than this value. If unspecified, at most 50 node groups are
      returned. The maximum value is 1000. Values greater than 1000 are forced
      to 1000.
    pageToken: A page token, received from a previous ListNodeGroups call.
      Provide this token to retrieve the subsequent page.When paginating, the
      other parameters provided to ListNodeGroups must match the call that
      provided the page token.
    parent: Required. The parent, which owns the collection of node groups.
      Format: projects/{project}/regions/{region}/clusters/{cluster}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
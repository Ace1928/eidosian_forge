from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsClustersListRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsClustersListRequest object.

  Fields:
    filter: Only resources matching this filter will be listed.
    orderBy: Specifies the order in which resources will be listed.
    pageSize: The maximum number of resources to list.
    pageToken: A page token received from previous list request. A page token
      received from previous list request.
    parent: Required. The parent location, which owns this collection of
      clusters.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
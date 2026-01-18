from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesListRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesListRequest object.

  Fields:
    filter: List filter.
    pageSize: Requested page size. Server may return fewer items than
      requested. If unspecified, the server will pick an appropriate default.
    pageToken: A token identifying a page of results from the server.
    parent: Required. Parent value for ListInstancesRequest.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagValuesListRequest(_messages.Message):
    """A CloudresourcemanagerTagValuesListRequest object.

  Fields:
    pageSize: Optional. The maximum number of TagValues to return in the
      response. The server allows a maximum of 300 TagValues to return. If
      unspecified, the server will use 100 as the default.
    pageToken: Optional. A pagination token returned from a previous call to
      `ListTagValues` that indicates where this listing should continue from.
    parent: Required. Resource name for the parent of the TagValues to be
      listed, in the format `tagKeys/123` or `tagValues/123`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3)
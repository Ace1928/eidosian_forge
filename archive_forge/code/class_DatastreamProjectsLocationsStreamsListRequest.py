from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsStreamsListRequest(_messages.Message):
    """A DatastreamProjectsLocationsStreamsListRequest object.

  Fields:
    filter: Filter request.
    orderBy: Order by fields for the result.
    pageSize: Maximum number of streams to return. If unspecified, at most 50
      streams will be returned. The maximum value is 1000; values above 1000
      will be coerced to 1000.
    pageToken: Page token received from a previous `ListStreams` call. Provide
      this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListStreams` must match the call that provided
      the page token.
    parent: Required. The parent that owns the collection of streams.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
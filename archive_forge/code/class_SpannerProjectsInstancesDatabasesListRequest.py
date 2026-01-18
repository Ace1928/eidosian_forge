from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesListRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesListRequest object.

  Fields:
    pageSize: Number of databases to be returned in the response. If 0 or
      less, defaults to the server's maximum allowed page size.
    pageToken: If non-empty, `page_token` should contain a next_page_token
      from a previous ListDatabasesResponse.
    parent: Required. The instance whose databases should be listed. Values
      are of the form `projects//instances/`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
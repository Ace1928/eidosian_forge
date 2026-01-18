from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsListRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsListRequest object.

  Fields:
    database: Required. The database in which to list sessions.
    filter: An expression for filtering the results of the request. Filter
      rules are case insensitive. The fields eligible for filtering are: *
      `labels.key` where key is the name of a label Some examples of using
      filters are: * `labels.env:*` --> The session has the label "env". *
      `labels.env:dev` --> The session has the label "env" and the value of
      the label contains the string "dev".
    pageSize: Number of sessions to be returned in the response. If 0 or less,
      defaults to the server's maximum allowed page size.
    pageToken: If non-empty, `page_token` should contain a next_page_token
      from a previous ListSessionsResponse.
  """
    database = _messages.StringField(1, required=True)
    filter = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
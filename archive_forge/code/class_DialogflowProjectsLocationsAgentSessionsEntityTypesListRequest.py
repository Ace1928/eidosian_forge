from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentSessionsEntityTypesListRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentSessionsEntityTypesListRequest object.

  Fields:
    pageSize: Optional. The maximum number of items to return in a single
      page. By default 100 and at most 1000.
    pageToken: Optional. The next_page_token value returned from a previous
      list request.
    parent: Required. The session to list all session entity types from.
      Format: `projects//agent/sessions/` or
      `projects//agent/environments//users// sessions/`. If `Environment ID`
      is not specified, we assume default 'draft' environment. If `User ID` is
      not specified, we assume default '-' user.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
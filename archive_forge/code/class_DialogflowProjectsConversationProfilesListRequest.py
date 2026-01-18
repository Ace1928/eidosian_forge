from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationProfilesListRequest(_messages.Message):
    """A DialogflowProjectsConversationProfilesListRequest object.

  Fields:
    pageSize: The maximum number of items to return in a single page. By
      default 100 and at most 1000.
    pageToken: The next_page_token value returned from a previous list
      request.
    parent: Required. The project to list all conversation profiles from.
      Format: `projects//locations/`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
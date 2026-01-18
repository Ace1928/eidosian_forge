from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationDatasetsListRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationDatasetsListRequest object.

  Fields:
    pageSize: Optional. Maximum number of conversation datasets to return in a
      single page. By default 100 and at most 1000.
    pageToken: Optional. The next_page_token value returned from a previous
      list request.
    parent: Required. The project and location name to list all conversation
      datasets for. Format: `projects//locations/`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
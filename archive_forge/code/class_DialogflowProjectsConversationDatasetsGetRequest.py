from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationDatasetsGetRequest(_messages.Message):
    """A DialogflowProjectsConversationDatasetsGetRequest object.

  Fields:
    name: Required. The conversation dataset to retrieve. Format:
      `projects//locations//conversationDatasets/`
  """
    name = _messages.StringField(1, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageQuickReplies(_messages.Message):
    """The quick replies response message.

  Fields:
    quickReplies: Optional. The collection of quick replies.
    title: Optional. The title of the collection of quick replies.
  """
    quickReplies = _messages.StringField(1, repeated=True)
    title = _messages.StringField(2)
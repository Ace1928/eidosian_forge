from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3InlineDestination(_messages.Message):
    """Inline destination for a Dialogflow operation that writes or exports
  objects (e.g. intents) outside of Dialogflow.

  Fields:
    content: Output only. The uncompressed byte content for the objects. Only
      populated in responses.
  """
    content = _messages.BytesField(1)
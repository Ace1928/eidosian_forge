from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ExportIntentsResponse(_messages.Message):
    """The response message for Intents.ExportIntents.

  Fields:
    intentsContent: Uncompressed byte content for intents. This field is
      populated only if `intents_content_inline` is set to true in
      ExportIntentsRequest.
    intentsUri: The URI to a file containing the exported intents. This field
      is populated only if `intents_uri` is specified in ExportIntentsRequest.
  """
    intentsContent = _messages.MessageField('GoogleCloudDialogflowCxV3InlineDestination', 1)
    intentsUri = _messages.StringField(2)
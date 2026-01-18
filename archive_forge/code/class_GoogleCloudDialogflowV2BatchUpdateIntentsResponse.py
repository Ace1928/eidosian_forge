from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchUpdateIntentsResponse(_messages.Message):
    """The response message for Intents.BatchUpdateIntents.

  Fields:
    intents: The collection of updated or created intents.
  """
    intents = _messages.MessageField('GoogleCloudDialogflowV2Intent', 1, repeated=True)
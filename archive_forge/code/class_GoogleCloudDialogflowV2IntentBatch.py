from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentBatch(_messages.Message):
    """This message is a wrapper around a collection of intents.

  Fields:
    intents: A collection of intents.
  """
    intents = _messages.MessageField('GoogleCloudDialogflowV2Intent', 1, repeated=True)
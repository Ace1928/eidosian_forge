from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchDeleteIntentsRequest(_messages.Message):
    """The request message for Intents.BatchDeleteIntents.

  Fields:
    intents: Required. The collection of intents to delete. Only intent `name`
      must be filled in.
  """
    intents = _messages.MessageField('GoogleCloudDialogflowV2Intent', 1, repeated=True)
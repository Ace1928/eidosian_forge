from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2EntityTypeBatch(_messages.Message):
    """This message is a wrapper around a collection of entity types.

  Fields:
    entityTypes: A collection of entity types.
  """
    entityTypes = _messages.MessageField('GoogleCloudDialogflowV2EntityType', 1, repeated=True)
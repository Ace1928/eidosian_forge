from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AddExecutionEventsRequest(_messages.Message):
    """Request message for MetadataService.AddExecutionEvents.

  Fields:
    events: The Events to create and add.
  """
    events = _messages.MessageField('GoogleCloudAiplatformV1beta1Event', 1, repeated=True)
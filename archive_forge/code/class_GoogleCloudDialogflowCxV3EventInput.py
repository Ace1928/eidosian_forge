from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3EventInput(_messages.Message):
    """Represents the event to trigger.

  Fields:
    event: Name of the event.
  """
    event = _messages.StringField(1)
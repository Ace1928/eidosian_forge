from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateTableMetadata(_messages.Message):
    """Metadata type for the operation returned by UpdateTable.

  Fields:
    endTime: If set, the time at which this operation finished or was
      canceled.
    name: The name of the table being updated.
    startTime: The time at which this operation started.
  """
    endTime = _messages.StringField(1)
    name = _messages.StringField(2)
    startTime = _messages.StringField(3)
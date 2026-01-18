from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StartBackfillJobResponse(_messages.Message):
    """Response for manually initiating a backfill job for a specific stream
  object.

  Fields:
    object: The stream object resource a backfill job was started for.
  """
    object = _messages.MessageField('StreamObject', 1)
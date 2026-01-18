from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobStatistics5(_messages.Message):
    """Statistics for a copy job.

  Fields:
    copiedLogicalBytes: Output only. Number of logical bytes copied to the
      destination table.
    copiedRows: Output only. Number of rows copied to the destination table.
  """
    copiedLogicalBytes = _messages.IntegerField(1)
    copiedRows = _messages.IntegerField(2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BinLogCoordinates(_messages.Message):
    """Binary log coordinates.

  Fields:
    binLogFileName: Name of the binary log file for a Cloud SQL instance.
    binLogPosition: Position (offset) within the binary log file.
    kind: This is always `sql#binLogCoordinates`.
  """
    binLogFileName = _messages.StringField(1)
    binLogPosition = _messages.IntegerField(2)
    kind = _messages.StringField(3)
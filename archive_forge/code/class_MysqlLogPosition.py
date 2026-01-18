from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MysqlLogPosition(_messages.Message):
    """MySQL log position

  Fields:
    logFile: Required. The binary log file name.
    logPosition: Optional. The position within the binary log file. Default is
      head of file.
  """
    logFile = _messages.StringField(1)
    logPosition = _messages.IntegerField(2, variant=_messages.Variant.INT32)
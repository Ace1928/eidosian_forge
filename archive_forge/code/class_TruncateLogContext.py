from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TruncateLogContext(_messages.Message):
    """Database Instance truncate log context.

  Fields:
    kind: This is always `sql#truncateLogContext`.
    logType: The type of log to truncate. Valid values are
      `MYSQL_GENERAL_TABLE` and `MYSQL_SLOW_TABLE`.
  """
    kind = _messages.StringField(1)
    logType = _messages.StringField(2)
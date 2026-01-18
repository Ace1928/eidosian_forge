from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesTruncateLogRequest(_messages.Message):
    """Instance truncate log request.

  Fields:
    truncateLogContext: Contains details about the truncate log operation.
  """
    truncateLogContext = _messages.MessageField('TruncateLogContext', 1)
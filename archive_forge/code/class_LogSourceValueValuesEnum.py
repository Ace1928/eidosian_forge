from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogSourceValueValuesEnum(_messages.Enum):
    """The source of log.

    Values:
      LOG_SOURCE_UNSPECIFIED: Unspecified source.
      TRAINING: Logs coming from Training dataset.
      SERVING: Logs coming from Serving traffic.
    """
    LOG_SOURCE_UNSPECIFIED = 0
    TRAINING = 1
    SERVING = 2
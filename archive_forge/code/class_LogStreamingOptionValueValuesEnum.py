from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogStreamingOptionValueValuesEnum(_messages.Enum):
    """Option to define build log streaming behavior to Cloud Storage.

    Values:
      STREAM_DEFAULT: Service may automatically determine build log streaming
        behavior.
      STREAM_ON: Build logs should be streamed to Cloud Storage.
      STREAM_OFF: Build logs should not be streamed to Cloud Storage; they
        will be written when the build is completed.
    """
    STREAM_DEFAULT = 0
    STREAM_ON = 1
    STREAM_OFF = 2
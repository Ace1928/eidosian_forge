from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationValueValuesEnum(_messages.Enum):
    """Where logs should be saved.

    Values:
      DESTINATION_UNSPECIFIED: Logs are not preserved.
      CLOUD_LOGGING: Logs are streamed to Cloud Logging.
      PATH: Logs are saved to a file path.
    """
    DESTINATION_UNSPECIFIED = 0
    CLOUD_LOGGING = 1
    PATH = 2
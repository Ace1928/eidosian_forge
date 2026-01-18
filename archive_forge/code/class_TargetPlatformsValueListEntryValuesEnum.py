from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TargetPlatformsValueListEntryValuesEnum(_messages.Enum):
    """TargetPlatformsValueListEntryValuesEnum enum type.

    Values:
      TARGET_PLATFORM_UNSPECIFIED: The target platform is unknown. Requests
        with this enum value will be rejected with INVALID_ARGUMENT error.
      APP_ENGINE: Google App Engine service.
      COMPUTE: Google Compute Engine service.
      CLOUD_RUN: Google Cloud Run service.
      CLOUD_FUNCTIONS: Google Cloud Function service.
    """
    TARGET_PLATFORM_UNSPECIFIED = 0
    APP_ENGINE = 1
    COMPUTE = 2
    CLOUD_RUN = 3
    CLOUD_FUNCTIONS = 4
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SdkSupportStatusValueValuesEnum(_messages.Enum):
    """The support status for this SDK version.

    Values:
      UNKNOWN: Dataflow is unaware of this version.
      SUPPORTED: This is a known version of an SDK, and is supported.
      STALE: A newer version of the SDK exists, and an update is recommended.
      DEPRECATED: This version of the SDK is deprecated and will eventually be
        unsupported.
      UNSUPPORTED: Support for this SDK version has ended and it should no
        longer be used.
    """
    UNKNOWN = 0
    SUPPORTED = 1
    STALE = 2
    DEPRECATED = 3
    UNSUPPORTED = 4
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdvertiseModeValueValuesEnum(_messages.Enum):
    """User-specified flag to indicate which mode to use for advertisement.

    Values:
      CUSTOM: <no description>
      DEFAULT: <no description>
    """
    CUSTOM = 0
    DEFAULT = 1
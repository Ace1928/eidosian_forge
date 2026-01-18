from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VersionStateValueValuesEnum(_messages.Enum):
    """Optional. Indicates the state of the model version.

    Values:
      VERSION_STATE_UNSPECIFIED: The version state is unspecified.
      VERSION_STATE_STABLE: Used to indicate the version is stable.
      VERSION_STATE_UNSTABLE: Used to indicate the version is unstable.
    """
    VERSION_STATE_UNSPECIFIED = 0
    VERSION_STATE_STABLE = 1
    VERSION_STATE_UNSTABLE = 2
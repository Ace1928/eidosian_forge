from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VersionPolicyValueValuesEnum(_messages.Enum):
    """Version policy defines the versions that the registry will accept.

    Values:
      VERSION_POLICY_UNSPECIFIED: VERSION_POLICY_UNSPECIFIED - the version
        policy is not defined. When the version policy is not defined, no
        validation is performed for the versions.
      RELEASE: RELEASE - repository will accept only Release versions.
      SNAPSHOT: SNAPSHOT - repository will accept only Snapshot versions.
    """
    VERSION_POLICY_UNSPECIFIED = 0
    RELEASE = 1
    SNAPSHOT = 2
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntitlementStateValueValuesEnum(_messages.Enum):
    """The current state of user's accessibility to a feature/benefit.

    Values:
      ENTITLEMENT_STATE_UNSPECIFIED: <no description>
      ENTITLED: User is entitled to a feature/benefit, but whether it has been
        successfully provisioned is decided by provisioning state.
      REVOKED: User is entitled to a feature/benefit, but it was requested to
        be revoked. Whether the revoke has been successful is decided by
        provisioning state.
    """
    ENTITLEMENT_STATE_UNSPECIFIED = 0
    ENTITLED = 1
    REVOKED = 2
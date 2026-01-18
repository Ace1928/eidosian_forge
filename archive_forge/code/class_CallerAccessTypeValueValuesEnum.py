from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CallerAccessTypeValueValuesEnum(_messages.Enum):
    """Required. Only Entitlements where the calling user has this access
    will be returned.

    Values:
      CALLER_ACCESS_TYPE_UNSPECIFIED: Unspecified access type.
      GRANT_REQUESTER: The user has access to create Grants using this
        Entitlement.
      GRANT_APPROVER: The user has access to approve/deny Grants created under
        this Entitlement.
    """
    CALLER_ACCESS_TYPE_UNSPECIFIED = 0
    GRANT_REQUESTER = 1
    GRANT_APPROVER = 2
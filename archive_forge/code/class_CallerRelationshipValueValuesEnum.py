from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CallerRelationshipValueValuesEnum(_messages.Enum):
    """Required. Only Grants which the caller is related to by this
    relationship will be returned in the response.

    Values:
      CALLER_RELATIONSHIP_TYPE_UNSPECIFIED: Unspecified caller relationship
        type.
      HAD_CREATED: The user had created this Grant by calling CreateGrant
        earlier.
      CAN_APPROVE: The user is an Approver for the Entitlement that this Grant
        is parented under and can currently approve/deny it.
      HAD_APPROVED: The caller had successfully approved/denied this Grant
        earlier.
    """
    CALLER_RELATIONSHIP_TYPE_UNSPECIFIED = 0
    HAD_CREATED = 1
    CAN_APPROVE = 2
    HAD_APPROVED = 3
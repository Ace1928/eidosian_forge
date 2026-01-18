from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttachStatusValueValuesEnum(_messages.Enum):
    """Status of the role attachment. Under development (go/si-attach-role),
    currently always return ATTACH_STATUS_UNSPECIFIED)

    Values:
      ATTACH_STATUS_UNSPECIFIED: Indicates that the AttachStatus was not set.
      ATTACHED: The default identity was attached to a role successfully in
        this request.
      ATTACH_SKIPPED: The request specified that no attempt should be made to
        attach the role.
      PREVIOUSLY_ATTACHED: Role was attached to the consumer project at some
        point in time. Tenant manager doesn't make assertion about the current
        state of the identity with respect to the consumer.  Role attachment
        should happen only once after activation and cannot be reattached
        after customer removes it. (go/si-attach-role)
      ATTACH_DENIED_BY_ORG_POLICY: Role attachment was denied in this request
        by customer set org policy. (go/si-attach-role)
    """
    ATTACH_STATUS_UNSPECIFIED = 0
    ATTACHED = 1
    ATTACH_SKIPPED = 2
    PREVIOUSLY_ATTACHED = 3
    ATTACH_DENIED_BY_ORG_POLICY = 4
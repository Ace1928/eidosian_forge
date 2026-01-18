from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DenyAccessStateValueValuesEnum(_messages.Enum):
    """Required. Indicates whether any policies attached to _this resource_
    deny the specific permission to the specified principal for the specified
    resource. This field does _not_ indicate whether the principal actually
    has the permission for the resource. There might be another policy that
    overrides this policy. To determine whether the principal actually has the
    permission, use the `overall_access_state` field in the
    TroubleshootIamPolicyResponse.

    Values:
      DENY_ACCESS_STATE_UNSPECIFIED: Not specified.
      DENY_ACCESS_STATE_DENIED: The deny policy denies the principal the
        permission.
      DENY_ACCESS_STATE_NOT_DENIED: The deny policy doesn't deny the principal
        the permission.
      DENY_ACCESS_STATE_UNKNOWN_CONDITIONAL: The deny policy denies the
        principal the permission if a condition expression evaluates to
        `true`. However, the sender of the request didn't provide enough
        context for Policy Troubleshooter to evaluate the condition
        expression.
      DENY_ACCESS_STATE_UNKNOWN_INFO: The sender of the request does not have
        access to all of the deny policies that Policy Troubleshooter needs to
        evaluate the principal's access.
    """
    DENY_ACCESS_STATE_UNSPECIFIED = 0
    DENY_ACCESS_STATE_DENIED = 1
    DENY_ACCESS_STATE_NOT_DENIED = 2
    DENY_ACCESS_STATE_UNKNOWN_CONDITIONAL = 3
    DENY_ACCESS_STATE_UNKNOWN_INFO = 4
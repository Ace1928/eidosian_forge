from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessValueValuesEnum(_messages.Enum):
    """Indicates whether _this policy_ provides the specified permission to
    the specified principal for the specified resource. This field does _not_
    indicate whether the principal actually has the permission for the
    resource. There might be another policy that overrides this policy. To
    determine whether the principal actually has the permission, use the
    `access` field in the TroubleshootIamPolicyResponse.

    Values:
      ACCESS_STATE_UNSPECIFIED: Default value. This value is unused.
      GRANTED: The principal has the permission.
      NOT_GRANTED: The principal does not have the permission.
      UNKNOWN_CONDITIONAL: The principal has the permission only if a
        condition expression evaluates to `true`.
      UNKNOWN_INFO_DENIED: The user who created the Replay does not have
        access to all of the policies that Policy Simulator needs to evaluate.
    """
    ACCESS_STATE_UNSPECIFIED = 0
    GRANTED = 1
    NOT_GRANTED = 2
    UNKNOWN_CONDITIONAL = 3
    UNKNOWN_INFO_DENIED = 4
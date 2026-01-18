from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuleAccessStateValueValuesEnum(_messages.Enum):
    """Output only. Indicates whether the rule allows access to the specified
    resource.

    Values:
      PAB_ACCESS_STATE_UNSPECIFIED: Not specified.
      PAB_ACCESS_STATE_ALLOWED: The PAB component allows the principal's
        access to the specified resource.
      PAB_ACCESS_STATE_NOT_ALLOWED: The PAB component doesn't allow the
        principal's access to the specified resource.
      PAB_ACCESS_STATE_NOT_ENFORCED: The PAB component is not enforced on the
        principal, or the specified resource. This state refers to 2 specific
        scenarios: - The service that the specified resource belongs to is not
        enforced by PAB at the policy version. - The binding doesn't apply to
        the principal, hence the policy is not enforced as a result.
      PAB_ACCESS_STATE_UNKNOWN_INFO: The sender of the request does not have
        access to the PAB component, or the relevant data to explain the PAB
        component.
    """
    PAB_ACCESS_STATE_UNSPECIFIED = 0
    PAB_ACCESS_STATE_ALLOWED = 1
    PAB_ACCESS_STATE_NOT_ALLOWED = 2
    PAB_ACCESS_STATE_NOT_ENFORCED = 3
    PAB_ACCESS_STATE_UNKNOWN_INFO = 4
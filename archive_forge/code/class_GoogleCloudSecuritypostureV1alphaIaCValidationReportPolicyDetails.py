from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritypostureV1alphaIaCValidationReportPolicyDetails(_messages.Message):
    """Details of policies unsupported by evaluation services during IAC
  validation.

  Enums:
    ConstraintTypeValueValuesEnum: Type of policy constraint.

  Fields:
    constraintType: Type of policy constraint.
    policyId: Policy id of unsupported policy.
  """

    class ConstraintTypeValueValuesEnum(_messages.Enum):
        """Type of policy constraint.

    Values:
      CONSTRAINT_TYPE_UNSPECIFIED: Unspecified constraint type.
      ORG_POLICY: Org policy canned constraint type.
      SECURITY_HEALTH_ANALYTICS_MODULE: SHA module canned constraint type.
      ORG_POLICY_CUSTOM: Custom org policy constraint type.
      SECURITY_HEALTH_ANALYTICS_CUSTOM_MODULE: Custom SHA constraint type.
    """
        CONSTRAINT_TYPE_UNSPECIFIED = 0
        ORG_POLICY = 1
        SECURITY_HEALTH_ANALYTICS_MODULE = 2
        ORG_POLICY_CUSTOM = 3
        SECURITY_HEALTH_ANALYTICS_CUSTOM_MODULE = 4
    constraintType = _messages.EnumField('ConstraintTypeValueValuesEnum', 1)
    policyId = _messages.StringField(2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyDetails(_messages.Message):
    """A PolicyDetails object.

  Enums:
    ConstraintTypeValueValuesEnum: Type of policy constraint.

  Fields:
    complianceStandards: Compliance standards that the policy maps to. E.g.
      CIS-2.0 1.15
    constraint: JSON string representing policy constraint.
      Format/representation may change, thus clients should not depend.
    constraintType: Type of policy constraint.
    description: Description of the policy.
  """

    class ConstraintTypeValueValuesEnum(_messages.Enum):
        """Type of policy constraint.

    Values:
      CONSTRAINT_TYPE_UNSPECIFIED: <no description>
      SECURITY_HEALTH_ANALYTICS_CUSTOM_MODULE: <no description>
      ORG_POLICY_CUSTOM: <no description>
      SECURITY_HEALTH_ANALYTICS_MODULE: SHA module constraint type.
      ORG_POLICY: Org policy constraint type.
    """
        CONSTRAINT_TYPE_UNSPECIFIED = 0
        SECURITY_HEALTH_ANALYTICS_CUSTOM_MODULE = 1
        ORG_POLICY_CUSTOM = 2
        SECURITY_HEALTH_ANALYTICS_MODULE = 3
        ORG_POLICY = 4
    complianceStandards = _messages.StringField(1, repeated=True)
    constraint = _messages.StringField(2)
    constraintType = _messages.EnumField('ConstraintTypeValueValuesEnum', 3)
    description = _messages.StringField(4)
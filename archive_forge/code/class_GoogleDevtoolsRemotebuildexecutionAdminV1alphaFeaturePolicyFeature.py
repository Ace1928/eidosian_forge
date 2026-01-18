from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature(_messages.Message):
    """Defines whether a feature can be used or what values are accepted.

  Enums:
    PolicyValueValuesEnum: The policy of the feature.

  Fields:
    allowedValues: A list of acceptable values. Only effective when the policy
      is `RESTRICTED`.
    policy: The policy of the feature.
  """

    class PolicyValueValuesEnum(_messages.Enum):
        """The policy of the feature.

    Values:
      POLICY_UNSPECIFIED: Default value, if not explicitly set. Equivalent to
        FORBIDDEN, unless otherwise documented on a specific Feature.
      ALLOWED: Feature is explicitly allowed.
      FORBIDDEN: Feature is forbidden. Requests attempting to leverage it will
        get an FailedPrecondition error, with a message like: "Feature
        forbidden by FeaturePolicy: Feature on instance "
      RESTRICTED: Only the values specified in the `allowed_values` are
        allowed.
    """
        POLICY_UNSPECIFIED = 0
        ALLOWED = 1
        FORBIDDEN = 2
        RESTRICTED = 3
    allowedValues = _messages.StringField(1, repeated=True)
    policy = _messages.EnumField('PolicyValueValuesEnum', 2)
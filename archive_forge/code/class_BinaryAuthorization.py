from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinaryAuthorization(_messages.Message):
    """Configuration for Binary Authorization.

  Enums:
    EvaluationModeValueValuesEnum: Mode of operation for binauthz policy
      evaluation. If unspecified, defaults to DISABLED.

  Fields:
    enabled: This field is deprecated. Leave this unset and instead configure
      BinaryAuthorization using evaluation_mode. If evaluation_mode is set to
      anything other than EVALUATION_MODE_UNSPECIFIED, this field is ignored.
    evaluationMode: Mode of operation for binauthz policy evaluation. If
      unspecified, defaults to DISABLED.
    policyBindings: Optional. Binauthz policies that apply to this cluster.
  """

    class EvaluationModeValueValuesEnum(_messages.Enum):
        """Mode of operation for binauthz policy evaluation. If unspecified,
    defaults to DISABLED.

    Values:
      EVALUATION_MODE_UNSPECIFIED: Default value
      DISABLED: Disable BinaryAuthorization
      PROJECT_SINGLETON_POLICY_ENFORCE: Enforce Kubernetes admission requests
        with BinaryAuthorization using the project's singleton policy. This is
        equivalent to setting the enabled boolean to true.
      POLICY_BINDINGS: Use Binary Authorization Continuous Validation with the
        policies specified in policy_bindings.
      POLICY_BINDINGS_AND_PROJECT_SINGLETON_POLICY_ENFORCE: Use Binary
        Authorization Continuous Validation with the policies specified in
        policy_bindings and enforce Kubernetes admission requests with Binary
        Authorization using the project's singleton policy.
    """
        EVALUATION_MODE_UNSPECIFIED = 0
        DISABLED = 1
        PROJECT_SINGLETON_POLICY_ENFORCE = 2
        POLICY_BINDINGS = 3
        POLICY_BINDINGS_AND_PROJECT_SINGLETON_POLICY_ENFORCE = 4
    enabled = _messages.BooleanField(1)
    evaluationMode = _messages.EnumField('EvaluationModeValueValuesEnum', 2)
    policyBindings = _messages.MessageField('PolicyBinding', 3, repeated=True)
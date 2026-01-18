from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadCertificateFeatureSpec(_messages.Message):
    """WorkloadCertificateFeatureSpec contains the input for the workload
  identity platform feature. This is required since Feature proto requires a
  spec.

  Enums:
    ProvisionGoogleCaValueValuesEnum: Immutable. Specifies CA configuration.

  Messages:
    MemberConfigsValue: Per-member configuration of workload certificate.

  Fields:
    defaultConfig: Default membership spec. Users can override the default in
      the member_configs for each member.
    memberConfigs: Per-member configuration of workload certificate.
    provisionGoogleCa: Immutable. Specifies CA configuration.
  """

    class ProvisionGoogleCaValueValuesEnum(_messages.Enum):
        """Immutable. Specifies CA configuration.

    Values:
      GOOGLE_CA_PROVISIONING_UNSPECIFIED: Disable default Google managed CA.
      DISABLED: Disable default Google managed CA.
      ENABLED: Use default Google managed CA.
      ENABLED_WITH_MANAGED_CA: Workload certificate feature is enabled, and
        the entire certificate provisioning process is managed by Google with
        managed CAS which is more secure than the default CA.
      ENABLED_WITH_DEFAULT_CA: Workload certificate feature is enabled, and
        the entire certificate provisioning process is using the default CA
        which is free.
    """
        GOOGLE_CA_PROVISIONING_UNSPECIFIED = 0
        DISABLED = 1
        ENABLED = 2
        ENABLED_WITH_MANAGED_CA = 3
        ENABLED_WITH_DEFAULT_CA = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MemberConfigsValue(_messages.Message):
        """Per-member configuration of workload certificate.

    Messages:
      AdditionalProperty: An additional property for a MemberConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MemberConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MemberConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A WorkloadCertificateMembershipSpec attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('WorkloadCertificateMembershipSpec', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    defaultConfig = _messages.MessageField('WorkloadCertificateMembershipSpec', 1)
    memberConfigs = _messages.MessageField('MemberConfigsValue', 2)
    provisionGoogleCa = _messages.EnumField('ProvisionGoogleCaValueValuesEnum', 3)
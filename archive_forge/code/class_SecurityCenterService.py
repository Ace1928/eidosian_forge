from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityCenterService(_messages.Message):
    """Represents a particular Security Command Center service. This includes
  settings information such as top-level enablement in addition to individual
  module settings. Service settings can be configured at the organization,
  folder, or project level. Service settings at the organization or folder
  level are inherited by those in child folders and projects.

  Enums:
    EffectiveEnablementStateValueValuesEnum: Output only. The effective
      enablement state for the service at its level of the resource hierarchy.
      If the intended state is set to INHERITED, the effective state will be
      inherited from the enablement state of an ancestor. This state may
      differ from the intended enablement state due to billing eligibility or
      onboarding status.
    IntendedEnablementStateValueValuesEnum: Optional. The intended state of
      enablement for the service at its level of the resource hierarchy. A
      DISABLED state will override all module enablement_states to DISABLED.

  Messages:
    ModulesValue: Optional. The configurations including the state of
      enablement for the service's different modules. The absence of a module
      in the map implies its configuration is inherited from its parent's.

  Fields:
    effectiveEnablementState: Output only. The effective enablement state for
      the service at its level of the resource hierarchy. If the intended
      state is set to INHERITED, the effective state will be inherited from
      the enablement state of an ancestor. This state may differ from the
      intended enablement state due to billing eligibility or onboarding
      status.
    intendedEnablementState: Optional. The intended state of enablement for
      the service at its level of the resource hierarchy. A DISABLED state
      will override all module enablement_states to DISABLED.
    modules: Optional. The configurations including the state of enablement
      for the service's different modules. The absence of a module in the map
      implies its configuration is inherited from its parent's.
    name: Identifier. The name of the service Formats: * organizations/{organi
      zation}/locations/{location}/securityCenterServices/{service} *
      folders/{folder}/locations/{location}/securityCenterServices/{service} *
      projects/{project}/locations/{location}/securityCenterServices/{service}
    updateTime: Output only. The time the service was last updated.
  """

    class EffectiveEnablementStateValueValuesEnum(_messages.Enum):
        """Output only. The effective enablement state for the service at its
    level of the resource hierarchy. If the intended state is set to
    INHERITED, the effective state will be inherited from the enablement state
    of an ancestor. This state may differ from the intended enablement state
    due to billing eligibility or onboarding status.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Default value. This value is unused.
      INHERITED: State is inherited from the parent resource. Not a valid
        effective enablement state.
      ENABLED: State is enabled.
      DISABLED: State is disabled.
    """
        ENABLEMENT_STATE_UNSPECIFIED = 0
        INHERITED = 1
        ENABLED = 2
        DISABLED = 3

    class IntendedEnablementStateValueValuesEnum(_messages.Enum):
        """Optional. The intended state of enablement for the service at its
    level of the resource hierarchy. A DISABLED state will override all module
    enablement_states to DISABLED.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Default value. This value is unused.
      INHERITED: State is inherited from the parent resource. Not a valid
        effective enablement state.
      ENABLED: State is enabled.
      DISABLED: State is disabled.
    """
        ENABLEMENT_STATE_UNSPECIFIED = 0
        INHERITED = 1
        ENABLED = 2
        DISABLED = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ModulesValue(_messages.Message):
        """Optional. The configurations including the state of enablement for the
    service's different modules. The absence of a module in the map implies
    its configuration is inherited from its parent's.

    Messages:
      AdditionalProperty: An additional property for a ModulesValue object.

    Fields:
      additionalProperties: Additional properties of type ModulesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ModulesValue object.

      Fields:
        key: Name of the additional property.
        value: A ModuleSettings attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ModuleSettings', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    effectiveEnablementState = _messages.EnumField('EffectiveEnablementStateValueValuesEnum', 1)
    intendedEnablementState = _messages.EnumField('IntendedEnablementStateValueValuesEnum', 2)
    modules = _messages.MessageField('ModulesValue', 3)
    name = _messages.StringField(4)
    updateTime = _messages.StringField(5)
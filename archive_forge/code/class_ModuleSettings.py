from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModuleSettings(_messages.Message):
    """The settings for individual modules.

  Enums:
    EffectiveEnablementStateValueValuesEnum: Output only. The effective
      enablement state for the module at its level of the resource hierarchy.
      If the intended state is set to INHERITED, the effective state will be
      inherited from the enablement state of an ancestor. This state may
      differ from the intended enablement state due to billing eligibility or
      onboarding status.
    IntendedEnablementStateValueValuesEnum: Optional. The intended state of
      enablement for the module at its level of the resource hierarchy.

  Fields:
    effectiveEnablementState: Output only. The effective enablement state for
      the module at its level of the resource hierarchy. If the intended state
      is set to INHERITED, the effective state will be inherited from the
      enablement state of an ancestor. This state may differ from the intended
      enablement state due to billing eligibility or onboarding status.
    intendedEnablementState: Optional. The intended state of enablement for
      the module at its level of the resource hierarchy.
  """

    class EffectiveEnablementStateValueValuesEnum(_messages.Enum):
        """Output only. The effective enablement state for the module at its
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
        """Optional. The intended state of enablement for the module at its level
    of the resource hierarchy.

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
    effectiveEnablementState = _messages.EnumField('EffectiveEnablementStateValueValuesEnum', 1)
    intendedEnablementState = _messages.EnumField('IntendedEnablementStateValueValuesEnum', 2)
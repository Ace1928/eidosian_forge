from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPostureConfig(_messages.Message):
    """SecurityPostureConfig defines the flags needed to enable/disable
  features for the Security Posture API.

  Enums:
    ModeValueValuesEnum: Sets which mode to use for Security Posture features.
    VulnerabilityModeValueValuesEnum: Sets which mode to use for vulnerability
      scanning.

  Fields:
    mode: Sets which mode to use for Security Posture features.
    vulnerabilityMode: Sets which mode to use for vulnerability scanning.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Sets which mode to use for Security Posture features.

    Values:
      MODE_UNSPECIFIED: Default value not specified.
      DISABLED: Disables Security Posture features on the cluster.
      BASIC: Applies Security Posture features on the cluster.
      ENTERPRISE: Applies the Security Posture off cluster Enterprise level
        features.
    """
        MODE_UNSPECIFIED = 0
        DISABLED = 1
        BASIC = 2
        ENTERPRISE = 3

    class VulnerabilityModeValueValuesEnum(_messages.Enum):
        """Sets which mode to use for vulnerability scanning.

    Values:
      VULNERABILITY_MODE_UNSPECIFIED: Default value not specified.
      VULNERABILITY_DISABLED: Disables vulnerability scanning on the cluster.
      VULNERABILITY_BASIC: Applies basic vulnerability scanning on the
        cluster.
      VULNERABILITY_ENTERPRISE: Applies the Security Posture's vulnerability
        on cluster Enterprise level features.
    """
        VULNERABILITY_MODE_UNSPECIFIED = 0
        VULNERABILITY_DISABLED = 1
        VULNERABILITY_BASIC = 2
        VULNERABILITY_ENTERPRISE = 3
    mode = _messages.EnumField('ModeValueValuesEnum', 1)
    vulnerabilityMode = _messages.EnumField('VulnerabilityModeValueValuesEnum', 2)
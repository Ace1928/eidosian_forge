from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2SensitiveDataProtectionMapping(_messages.Message):
    """Resource value mapping for Sensitive Data Protection findings If any of
  these mappings have a resource value that is not unspecified, the
  resource_value field will be ignored when reading this configuration.

  Enums:
    HighSensitivityMappingValueValuesEnum: Resource value mapping for high-
      sensitivity Sensitive Data Protection findings
    MediumSensitivityMappingValueValuesEnum: Resource value mapping for
      medium-sensitivity Sensitive Data Protection findings

  Fields:
    highSensitivityMapping: Resource value mapping for high-sensitivity
      Sensitive Data Protection findings
    mediumSensitivityMapping: Resource value mapping for medium-sensitivity
      Sensitive Data Protection findings
  """

    class HighSensitivityMappingValueValuesEnum(_messages.Enum):
        """Resource value mapping for high-sensitivity Sensitive Data Protection
    findings

    Values:
      RESOURCE_VALUE_UNSPECIFIED: Unspecific value
      HIGH: High resource value
      MEDIUM: Medium resource value
      LOW: Low resource value
      NONE: No resource value, e.g. ignore these resources
    """
        RESOURCE_VALUE_UNSPECIFIED = 0
        HIGH = 1
        MEDIUM = 2
        LOW = 3
        NONE = 4

    class MediumSensitivityMappingValueValuesEnum(_messages.Enum):
        """Resource value mapping for medium-sensitivity Sensitive Data
    Protection findings

    Values:
      RESOURCE_VALUE_UNSPECIFIED: Unspecific value
      HIGH: High resource value
      MEDIUM: Medium resource value
      LOW: Low resource value
      NONE: No resource value, e.g. ignore these resources
    """
        RESOURCE_VALUE_UNSPECIFIED = 0
        HIGH = 1
        MEDIUM = 2
        LOW = 3
        NONE = 4
    highSensitivityMapping = _messages.EnumField('HighSensitivityMappingValueValuesEnum', 1)
    mediumSensitivityMapping = _messages.EnumField('MediumSensitivityMappingValueValuesEnum', 2)
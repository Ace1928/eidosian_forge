from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyDdosProtectionConfig(_messages.Message):
    """A SecurityPolicyDdosProtectionConfig object.

  Enums:
    DdosProtectionValueValuesEnum:

  Fields:
    ddosProtection: A DdosProtectionValueValuesEnum attribute.
  """

    class DdosProtectionValueValuesEnum(_messages.Enum):
        """DdosProtectionValueValuesEnum enum type.

    Values:
      ADVANCED: <no description>
      ADVANCED_PREVIEW: <no description>
      STANDARD: <no description>
    """
        ADVANCED = 0
        ADVANCED_PREVIEW = 1
        STANDARD = 2
    ddosProtection = _messages.EnumField('DdosProtectionValueValuesEnum', 1)
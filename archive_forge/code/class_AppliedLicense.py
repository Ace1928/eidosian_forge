from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppliedLicense(_messages.Message):
    """AppliedLicense holds the license data returned by adaptation module
  report.

  Enums:
    TypeValueValuesEnum: The license type that was used in OS adaptation.

  Fields:
    osLicense: The OS license returned from the adaptation module's report.
    type: The license type that was used in OS adaptation.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The license type that was used in OS adaptation.

    Values:
      TYPE_UNSPECIFIED: Unspecified license for the OS.
      NONE: No license available for the OS.
      PAYG: The license type is Pay As You Go license type.
      BYOL: The license type is Bring Your Own License type.
    """
        TYPE_UNSPECIFIED = 0
        NONE = 1
        PAYG = 2
        BYOL = 3
    osLicense = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)
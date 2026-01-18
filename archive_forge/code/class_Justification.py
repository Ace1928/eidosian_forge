from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Justification(_messages.Message):
    """Justification provides the justification when the state of the
  assessment if NOT_AFFECTED.

  Enums:
    JustificationTypeValueValuesEnum: The justification type for this
      vulnerability.

  Fields:
    details: Additional details on why this justification was chosen.
    justificationType: The justification type for this vulnerability.
  """

    class JustificationTypeValueValuesEnum(_messages.Enum):
        """The justification type for this vulnerability.

    Values:
      JUSTIFICATION_TYPE_UNSPECIFIED: JUSTIFICATION_TYPE_UNSPECIFIED.
      COMPONENT_NOT_PRESENT: The vulnerable component is not present in the
        product.
      VULNERABLE_CODE_NOT_PRESENT: The vulnerable code is not present.
        Typically this case occurs when source code is configured or built in
        a way that excludes the vulnerable code.
      VULNERABLE_CODE_NOT_IN_EXECUTE_PATH: The vulnerable code can not be
        executed. Typically this case occurs when the product includes the
        vulnerable code but does not call or use the vulnerable code.
      VULNERABLE_CODE_CANNOT_BE_CONTROLLED_BY_ADVERSARY: The vulnerable code
        cannot be controlled by an attacker to exploit the vulnerability.
      INLINE_MITIGATIONS_ALREADY_EXIST: The product includes built-in
        protections or features that prevent exploitation of the
        vulnerability. These built-in protections cannot be subverted by the
        attacker and cannot be configured or disabled by the user. These
        mitigations completely prevent exploitation based on known attack
        vectors.
    """
        JUSTIFICATION_TYPE_UNSPECIFIED = 0
        COMPONENT_NOT_PRESENT = 1
        VULNERABLE_CODE_NOT_PRESENT = 2
        VULNERABLE_CODE_NOT_IN_EXECUTE_PATH = 3
        VULNERABLE_CODE_CANNOT_BE_CONTROLLED_BY_ADVERSARY = 4
        INLINE_MITIGATIONS_ALREADY_EXIST = 5
    details = _messages.StringField(1)
    justificationType = _messages.EnumField('JustificationTypeValueValuesEnum', 2)
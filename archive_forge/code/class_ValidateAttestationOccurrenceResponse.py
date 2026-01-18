from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ValidateAttestationOccurrenceResponse(_messages.Message):
    """Response message for ValidationHelperV1.ValidateAttestationOccurrence.

  Enums:
    ResultValueValuesEnum: The result of the Attestation validation.

  Fields:
    denialReason: The reason for denial if the Attestation couldn't be
      validated.
    result: The result of the Attestation validation.
  """

    class ResultValueValuesEnum(_messages.Enum):
        """The result of the Attestation validation.

    Values:
      RESULT_UNSPECIFIED: Unspecified.
      VERIFIED: The Attestation was able to verified by the Attestor.
      ATTESTATION_NOT_VERIFIABLE: The Attestation was not able to verified by
        the Attestor.
    """
        RESULT_UNSPECIFIED = 0
        VERIFIED = 1
        ATTESTATION_NOT_VERIFIABLE = 2
    denialReason = _messages.StringField(1)
    result = _messages.EnumField('ResultValueValuesEnum', 2)
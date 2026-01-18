from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ValidateAttestationOccurrenceRequest(_messages.Message):
    """Request message for ValidationHelperV1.ValidateAttestationOccurrence.

  Fields:
    attestation: Required. An AttestationOccurrence to be checked that it can
      be verified by the `Attestor`. It does not have to be an existing entity
      in Container Analysis. It must otherwise be a valid
      `AttestationOccurrence`.
    occurrenceNote: Required. The resource name of the Note to which the
      containing Occurrence is associated.
    occurrenceResourceUri: Required. The URI of the artifact (e.g. container
      image) that is the subject of the containing Occurrence.
  """
    attestation = _messages.MessageField('AttestationOccurrence', 1)
    occurrenceNote = _messages.StringField(2)
    occurrenceResourceUri = _messages.StringField(3)
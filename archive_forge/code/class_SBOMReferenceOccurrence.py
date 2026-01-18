from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SBOMReferenceOccurrence(_messages.Message):
    """The occurrence representing an SBOM reference as applied to a specific
  resource. The occurrence follows the DSSE specification. See
  https://github.com/secure-systems-lab/dsse/blob/master/envelope.md for more
  details.

  Fields:
    payload: The actual payload that contains the SBOM reference data.
    payloadType: The kind of payload that SbomReferenceIntotoPayload takes.
      Since it's in the intoto format, this value is expected to be
      'application/vnd.in-toto+json'.
    signatures: The signatures over the payload.
  """
    payload = _messages.MessageField('SbomReferenceIntotoPayload', 1)
    payloadType = _messages.StringField(2)
    signatures = _messages.MessageField('EnvelopeSignature', 3, repeated=True)
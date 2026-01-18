from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttestationOccurrence(_messages.Message):
    """Occurrence that represents a single "attestation". The authenticity of
  an attestation can be verified using the attached signature. If the verifier
  trusts the public key of the signer, then verifying the signature is
  sufficient to establish trust. In this circumstance, the authority to which
  this attestation is attached is primarily useful for lookup (how to find
  this attestation if you already know the authority and artifact to be
  verified) and intent (for which authority this attestation was intended to
  sign.

  Fields:
    jwts: One or more JWTs encoding a self-contained attestation. Each JWT
      encodes the payload that it verifies within the JWT itself. Verifier
      implementation SHOULD ignore the `serialized_payload` field when
      verifying these JWTs. If only JWTs are present on this
      AttestationOccurrence, then the `serialized_payload` SHOULD be left
      empty. Each JWT SHOULD encode a claim specific to the `resource_uri` of
      this Occurrence, but this is not validated by Grafeas metadata API
      implementations. The JWT itself is opaque to Grafeas.
    serializedPayload: Required. The serialized payload that is verified by
      one or more `signatures`.
    signatures: One or more signatures over `serialized_payload`. Verifier
      implementations should consider this attestation message verified if at
      least one `signature` verifies `serialized_payload`. See `Signature` in
      common.proto for more details on signature structure and verification.
  """
    jwts = _messages.MessageField('Jwt', 1, repeated=True)
    serializedPayload = _messages.BytesField(2)
    signatures = _messages.MessageField('Signature', 3, repeated=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenericSignedAttestation(_messages.Message):
    """An attestation wrapper that uses the Grafeas `Signature` message. This
  attestation must define the `serialized_payload` that the `signatures`
  verify and any metadata necessary to interpret that plaintext. The
  signatures should always be over the `serialized_payload` bytestring.

  Enums:
    ContentTypeValueValuesEnum: Type (for example schema) of the attestation
      payload that was signed. The verifier must ensure that the provided type
      is one that the verifier supports, and that the attestation payload is a
      valid instantiation of that type (for example by validating a JSON
      schema).

  Fields:
    contentType: Type (for example schema) of the attestation payload that was
      signed. The verifier must ensure that the provided type is one that the
      verifier supports, and that the attestation payload is a valid
      instantiation of that type (for example by validating a JSON schema).
    serializedPayload: The serialized payload that is verified by one or more
      `signatures`. The encoding and semantic meaning of this payload must
      match what is set in `content_type`.
    signatures: One or more signatures over `serialized_payload`. Verifier
      implementations should consider this attestation message verified if at
      least one `signature` verifies `serialized_payload`. See `Signature` in
      common.proto for more details on signature structure and verification.
  """

    class ContentTypeValueValuesEnum(_messages.Enum):
        """Type (for example schema) of the attestation payload that was signed.
    The verifier must ensure that the provided type is one that the verifier
    supports, and that the attestation payload is a valid instantiation of
    that type (for example by validating a JSON schema).

    Values:
      CONTENT_TYPE_UNSPECIFIED: `ContentType` is not set.
      SIMPLE_SIGNING_JSON: Atomic format attestation signature. See https://gi
        thub.com/containers/image/blob/8a5d2f82a6e3263290c8e0276c3e0f64e77723e
        7/docs/atomic-signature.md The payload extracted in `plaintext` is a
        JSON blob conforming to the linked schema.
    """
        CONTENT_TYPE_UNSPECIFIED = 0
        SIMPLE_SIGNING_JSON = 1
    contentType = _messages.EnumField('ContentTypeValueValuesEnum', 1)
    serializedPayload = _messages.BytesField(2)
    signatures = _messages.MessageField('Signature', 3, repeated=True)
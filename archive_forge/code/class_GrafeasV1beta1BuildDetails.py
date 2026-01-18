from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1beta1BuildDetails(_messages.Message):
    """Details of a build occurrence.

  Fields:
    inTotoSlsaProvenanceV1: A InTotoSlsaProvenanceV1 attribute.
    provenance: Required. The actual provenance for the build.
    provenanceBytes: Serialized JSON representation of the provenance, used in
      generating the build signature in the corresponding build note. After
      verifying the signature, `provenance_bytes` can be unmarshalled and
      compared to the provenance to confirm that it is unchanged. A
      base64-encoded string representation of the provenance bytes is used for
      the signature in order to interoperate with openssl which expects this
      format for signature verification. The serialized form is captured both
      to avoid ambiguity in how the provenance is marshalled to json as well
      to prevent incompatibilities with future changes.
  """
    inTotoSlsaProvenanceV1 = _messages.MessageField('InTotoSlsaProvenanceV1', 1)
    provenance = _messages.MessageField('BuildProvenance', 2)
    provenanceBytes = _messages.StringField(3)
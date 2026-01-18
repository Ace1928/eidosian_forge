from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildOccurrence(_messages.Message):
    """Details of a build occurrence.

  Fields:
    inTotoSlsaProvenanceV1: In-Toto Slsa Provenance V1 represents a slsa
      provenance meeting the slsa spec, wrapped in an in-toto statement. This
      allows for direct jsonification of a to-spec in-toto slsa statement with
      a to-spec slsa provenance.
    intotoProvenance: Deprecated. See InTotoStatement for the replacement. In-
      toto Provenance representation as defined in spec.
    intotoStatement: In-toto Statement representation as defined in spec. The
      intoto_statement can contain any type of provenance. The serialized
      payload of the statement can be stored and signed in the Occurrence's
      envelope.
    provenance: The actual provenance for the build.
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
    intotoProvenance = _messages.MessageField('InTotoProvenance', 2)
    intotoStatement = _messages.MessageField('InTotoStatement', 3)
    provenance = _messages.MessageField('BuildProvenance', 4)
    provenanceBytes = _messages.StringField(5)
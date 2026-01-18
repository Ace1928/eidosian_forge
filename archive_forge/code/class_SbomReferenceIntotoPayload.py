from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SbomReferenceIntotoPayload(_messages.Message):
    """The actual payload that contains the SBOM Reference data. The payload
  follows the intoto statement specification. See https://github.com/in-
  toto/attestation/blob/main/spec/v1.0/statement.md for more details.

  Fields:
    _type: Identifier for the schema of the Statement.
    predicate: Additional parameters of the Predicate. Includes the actual
      data about the SBOM.
    predicateType: URI identifying the type of the Predicate.
    subject: Set of software artifacts that the attestation applies to. Each
      element represents a single software artifact.
  """
    _type = _messages.StringField(1)
    predicate = _messages.MessageField('SbomReferenceIntotoPredicate', 2)
    predicateType = _messages.StringField(3)
    subject = _messages.MessageField('Subject', 4, repeated=True)
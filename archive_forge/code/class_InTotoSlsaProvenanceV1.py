from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InTotoSlsaProvenanceV1(_messages.Message):
    """A InTotoSlsaProvenanceV1 object.

  Fields:
    _type: InToto spec defined at https://github.com/in-
      toto/attestation/tree/main/spec#statement
    predicate: A SlsaProvenanceV1 attribute.
    predicateType: A string attribute.
    subject: A Subject attribute.
  """
    _type = _messages.StringField(1)
    predicate = _messages.MessageField('SlsaProvenanceV1', 2)
    predicateType = _messages.StringField(3)
    subject = _messages.MessageField('Subject', 4, repeated=True)
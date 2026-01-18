from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SbomReferenceIntotoPredicate(_messages.Message):
    """A predicate which describes the SBOM being referenced.

  Messages:
    DigestValue: A map of algorithm to digest of the contents of the SBOM.

  Fields:
    digest: A map of algorithm to digest of the contents of the SBOM.
    location: The location of the SBOM.
    mimeType: The mime type of the SBOM.
    referrerId: The person or system referring this predicate to the consumer.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DigestValue(_messages.Message):
        """A map of algorithm to digest of the contents of the SBOM.

    Messages:
      AdditionalProperty: An additional property for a DigestValue object.

    Fields:
      additionalProperties: Additional properties of type DigestValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DigestValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    digest = _messages.MessageField('DigestValue', 1)
    location = _messages.StringField(2)
    mimeType = _messages.StringField(3)
    referrerId = _messages.StringField(4)
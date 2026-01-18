from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1SlsaProvenanceZeroTwoSlsaMaterial(_messages.Message):
    """The collection of artifacts that influenced the build including sources,
  dependencies, build tools, base images, and so on.

  Messages:
    DigestValue: A DigestValue object.

  Fields:
    digest: A DigestValue attribute.
    uri: A string attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DigestValue(_messages.Message):
        """A DigestValue object.

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
    uri = _messages.StringField(2)
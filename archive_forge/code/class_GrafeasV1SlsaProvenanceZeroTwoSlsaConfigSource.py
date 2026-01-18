from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1SlsaProvenanceZeroTwoSlsaConfigSource(_messages.Message):
    """Describes where the config file that kicked off the build came from.
  This is effectively a pointer to the source where buildConfig came from.

  Messages:
    DigestValue: A DigestValue object.

  Fields:
    digest: A DigestValue attribute.
    entryPoint: A string attribute.
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
    entryPoint = _messages.StringField(2)
    uri = _messages.StringField(3)
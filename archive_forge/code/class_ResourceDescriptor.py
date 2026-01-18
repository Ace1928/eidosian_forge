from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceDescriptor(_messages.Message):
    """A ResourceDescriptor object.

  Messages:
    AnnotationsValue: A AnnotationsValue object.
    DigestValue: A DigestValue object.

  Fields:
    annotations: A AnnotationsValue attribute.
    content: A byte attribute.
    digest: A DigestValue attribute.
    downloadLocation: A string attribute.
    mediaType: A string attribute.
    name: A string attribute.
    uri: A string attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """A AnnotationsValue object.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

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
    annotations = _messages.MessageField('AnnotationsValue', 1)
    content = _messages.BytesField(2)
    digest = _messages.MessageField('DigestValue', 3)
    downloadLocation = _messages.StringField(4)
    mediaType = _messages.StringField(5)
    name = _messages.StringField(6)
    uri = _messages.StringField(7)
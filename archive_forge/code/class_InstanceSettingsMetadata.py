from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceSettingsMetadata(_messages.Message):
    """A InstanceSettingsMetadata object.

  Messages:
    ItemsValue: A metadata key/value items map. The total size of all keys and
      values must be less than 512KB.

  Fields:
    items: A metadata key/value items map. The total size of all keys and
      values must be less than 512KB.
    kind: [Output Only] Type of the resource. Always compute#metadata for
      metadata.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ItemsValue(_messages.Message):
        """A metadata key/value items map. The total size of all keys and values
    must be less than 512KB.

    Messages:
      AdditionalProperty: An additional property for a ItemsValue object.

    Fields:
      additionalProperties: Additional properties of type ItemsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ItemsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    items = _messages.MessageField('ItemsValue', 1)
    kind = _messages.StringField(2, default='compute#metadata')
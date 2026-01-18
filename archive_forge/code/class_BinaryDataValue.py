from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class BinaryDataValue(_messages.Message):
    """BinaryData contains the binary data. Each key must consist of
    alphanumeric characters, '-', '_' or '.'. BinaryData can contain byte
    sequences that are not in the UTF-8 range. The keys stored in BinaryData
    must not overlap with the ones in the Data field, this is enforced during
    validation process. Using this field will require 1.10+ apiserver and
    kubelet.

    Messages:
      AdditionalProperty: An additional property for a BinaryDataValue object.

    Fields:
      additionalProperties: Additional properties of type BinaryDataValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a BinaryDataValue object.

      Fields:
        key: Name of the additional property.
        value: A byte attribute.
      """
        key = _messages.StringField(1)
        value = _messages.BytesField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
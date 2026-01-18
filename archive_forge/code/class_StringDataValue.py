from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class StringDataValue(_messages.Message):
    """stringData allows specifying non-binary secret data in string form. It
    is provided as a write-only convenience method. All keys and values are
    merged into the data field on write, overwriting any existing values. It
    is never output when reading from the API. +k8s:conversion-gen=false

    Messages:
      AdditionalProperty: An additional property for a StringDataValue object.

    Fields:
      additionalProperties: Additional properties of type StringDataValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a StringDataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
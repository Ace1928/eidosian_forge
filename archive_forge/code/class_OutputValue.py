from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class OutputValue(_messages.Message):
    """Mapping of the output parameter name to its output definition. Will be
    replacing outputs in future.

    Messages:
      AdditionalProperty: An additional property for a OutputValue object.

    Fields:
      additionalProperties: Additional properties of type OutputValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a OutputValue object.

      Fields:
        key: Name of the additional property.
        value: A Output attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Output', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class InputValuesValue(_messages.Message):
    """Input variable values for the Terraform blueprint.

    Messages:
      AdditionalProperty: An additional property for a InputValuesValue
        object.

    Fields:
      additionalProperties: Additional properties of type InputValuesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a InputValuesValue object.

      Fields:
        key: Name of the additional property.
        value: A TerraformVariable attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('TerraformVariable', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
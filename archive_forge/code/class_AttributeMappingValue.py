from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AttributeMappingValue(_messages.Message):
    """Optional. The mapping of additional user attributes like nickname,
    birthday and address etc.. `key` is the name of this additional attribute.
    `value` is a string presenting as CEL(common expression language, go/cel)
    used for getting the value from the resources. Take nickname as an
    example, in this case, `key` is "attribute.nickname" and `value` is
    "assertion.nickname".

    Messages:
      AdditionalProperty: An additional property for a AttributeMappingValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        AttributeMappingValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AttributeMappingValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
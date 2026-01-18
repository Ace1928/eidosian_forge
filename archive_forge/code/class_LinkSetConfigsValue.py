from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LinkSetConfigsValue(_messages.Message):
    """Mapping of a collection of link sets to the set configuration.

    Messages:
      AdditionalProperty: An additional property for a LinkSetConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type LinkSetConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LinkSetConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A LinkSetConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('LinkSetConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
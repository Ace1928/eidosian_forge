from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LinkConfigsValue(_messages.Message):
    """Mapping of a link field name to its configuration.

    Messages:
      AdditionalProperty: An additional property for a LinkConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type LinkConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LinkConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A LinkConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('LinkConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
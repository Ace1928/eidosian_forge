from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LinkInfoValue(_messages.Message):
    """Information about the links.

    Messages:
      AdditionalProperty: An additional property for a LinkInfoValue object.

    Fields:
      additionalProperties: Additional properties of type LinkInfoValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LinkInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A LinkInfo attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('LinkInfo', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
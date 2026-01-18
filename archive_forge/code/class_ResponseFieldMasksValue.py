from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ResponseFieldMasksValue(_messages.Message):
    """Defines which part of the response a child operation will contribute.
    Each key of the map is the name of a child operation. Each value is a
    field mask that identifies what that child operation contributes to the
    response, for example, "quota_settings", "visiblity_settings", etc.

    Messages:
      AdditionalProperty: An additional property for a ResponseFieldMasksValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ResponseFieldMasksValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ResponseFieldMasksValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
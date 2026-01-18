from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ErrorsValue(_messages.Message):
    """Maps the name of each tagged column (or empty string for a sole entry)
    to tagging operation status.

    Messages:
      AdditionalProperty: An additional property for a ErrorsValue object.

    Fields:
      additionalProperties: Additional properties of type ErrorsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ErrorsValue object.

      Fields:
        key: Name of the additional property.
        value: A Status attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Status', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
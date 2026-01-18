from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocalizedString(_messages.Message):
    """A message representing a user-facing string whose value may need to be
  translated before being displayed.

  Messages:
    ArgsValue: A map of arguments used when creating the localized message.
      Keys represent parameter names which may be used by the localized
      version when substituting dynamic values.

  Fields:
    args: A map of arguments used when creating the localized message. Keys
      represent parameter names which may be used by the localized version
      when substituting dynamic values.
    message: The canonical English version of this message. If no token is
      provided or the front-end has no message associated with the token, this
      text will be displayed as-is.
    token: The token identifying the message, e.g. 'METRIC_READ_CPU'. This
      should be unique within the service.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ArgsValue(_messages.Message):
        """A map of arguments used when creating the localized message. Keys
    represent parameter names which may be used by the localized version when
    substituting dynamic values.

    Messages:
      AdditionalProperty: An additional property for a ArgsValue object.

    Fields:
      additionalProperties: Additional properties of type ArgsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ArgsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    args = _messages.MessageField('ArgsValue', 1)
    message = _messages.StringField(2)
    token = _messages.StringField(3)
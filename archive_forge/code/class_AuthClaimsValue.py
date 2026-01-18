from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AuthClaimsValue(_messages.Message):
    """Evaluate the auth policy with a customized JWT auth token. Should
    follow the Firebase Auth token format.
    https://firebase.google.com/docs/rules/rules-and-auth For example: a
    verified user may have auth_claims of {"sub": , "email_verified": true}

    Messages:
      AdditionalProperty: An additional property for a AuthClaimsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AuthClaimsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoadInstanceAuthInfoResponse(_messages.Message):
    """Response for LoadInstanceAuthInfo.

  Messages:
    UserAccountsValue: Map of username to the user account info.

  Fields:
    sshKeys: List of ssh keys.
    userAccounts: Map of username to the user account info.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserAccountsValue(_messages.Message):
        """Map of username to the user account info.

    Messages:
      AdditionalProperty: An additional property for a UserAccountsValue
        object.

    Fields:
      additionalProperties: Additional properties of type UserAccountsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserAccountsValue object.

      Fields:
        key: Name of the additional property.
        value: A UserAccount attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('UserAccount', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    sshKeys = _messages.MessageField('SSHKey', 1, repeated=True)
    userAccounts = _messages.MessageField('UserAccountsValue', 2)
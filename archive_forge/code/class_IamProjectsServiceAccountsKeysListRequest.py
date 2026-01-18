from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamProjectsServiceAccountsKeysListRequest(_messages.Message):
    """A IamProjectsServiceAccountsKeysListRequest object.

  Enums:
    KeyTypesValueValuesEnum: Filters the types of keys the user wants to
      include in the list response. Duplicate key types are not allowed. If no
      key type is provided, all keys are returned.

  Fields:
    keyTypes: Filters the types of keys the user wants to include in the list
      response. Duplicate key types are not allowed. If no key type is
      provided, all keys are returned.
    name: The resource name of the service account in the following format:
      `projects/{project}/serviceAccounts/{account}`.  Using `-` as a wildcard
      for the project, will infer the project from the account. The `account`
      value can be the `email` address or the `unique_id` of the service
      account.
  """

    class KeyTypesValueValuesEnum(_messages.Enum):
        """Filters the types of keys the user wants to include in the list
    response. Duplicate key types are not allowed. If no key type is provided,
    all keys are returned.

    Values:
      KEY_TYPE_UNSPECIFIED: <no description>
      USER_MANAGED: <no description>
      SYSTEM_MANAGED: <no description>
    """
        KEY_TYPE_UNSPECIFIED = 0
        USER_MANAGED = 1
        SYSTEM_MANAGED = 2
    keyTypes = _messages.EnumField('KeyTypesValueValuesEnum', 1, repeated=True)
    name = _messages.StringField(2, required=True)
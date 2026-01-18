from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamProjectsServiceAccountsKeysGetRequest(_messages.Message):
    """A IamProjectsServiceAccountsKeysGetRequest object.

  Enums:
    PublicKeyTypeValueValuesEnum: The output format of the public key
      requested. X509_PEM is the default output format.

  Fields:
    name: The resource name of the service account key in the following
      format: `projects/{project}/serviceAccounts/{account}/keys/{key}`.
      Using `-` as a wildcard for the project will infer the project from the
      account. The `account` value can be the `email` address or the
      `unique_id` of the service account.
    publicKeyType: The output format of the public key requested. X509_PEM is
      the default output format.
  """

    class PublicKeyTypeValueValuesEnum(_messages.Enum):
        """The output format of the public key requested. X509_PEM is the default
    output format.

    Values:
      TYPE_NONE: <no description>
      TYPE_X509_PEM_FILE: <no description>
      TYPE_RAW_PUBLIC_KEY: <no description>
    """
        TYPE_NONE = 0
        TYPE_X509_PEM_FILE = 1
        TYPE_RAW_PUBLIC_KEY = 2
    name = _messages.StringField(1, required=True)
    publicKeyType = _messages.EnumField('PublicKeyTypeValueValuesEnum', 2)
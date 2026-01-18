from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ServiceAccountKey(_messages.Message):
    """Represents a service account key.  A service account has two sets of
  key-pairs: user-managed, and system-managed.  User-managed key-pairs can be
  created and deleted by users.  Users are responsible for rotating these keys
  periodically to ensure security of their service accounts.  Users retain the
  private key of these key-pairs, and Google retains ONLY the public key.
  System-managed key-pairs are managed automatically by Google, and rotated
  daily without user intervention.  The private key never leaves Google's
  servers to maximize security.  Public keys for all service accounts are also
  published at the OAuth2 Service Account API.

  Enums:
    PrivateKeyTypeValueValuesEnum: The output format for the private key. Only
      provided in `CreateServiceAccountKey` responses, not in
      `GetServiceAccountKey` or `ListServiceAccountKey` responses.  Google
      never exposes system-managed private keys, and never retains user-
      managed private keys.

  Fields:
    name: The resource name of the service account key in the following format
      `projects/{project}/serviceAccounts/{account}/keys/{key}`.
    privateKeyData: The private key data. Only provided in
      `CreateServiceAccountKey` responses.
    privateKeyType: The output format for the private key. Only provided in
      `CreateServiceAccountKey` responses, not in `GetServiceAccountKey` or
      `ListServiceAccountKey` responses.  Google never exposes system-managed
      private keys, and never retains user-managed private keys.
    publicKeyData: The public key data. Only provided in
      `GetServiceAccountKey` responses.
    validAfterTime: The key can be used after this timestamp.
    validBeforeTime: The key can be used before this timestamp.
  """

    class PrivateKeyTypeValueValuesEnum(_messages.Enum):
        """The output format for the private key. Only provided in
    `CreateServiceAccountKey` responses, not in `GetServiceAccountKey` or
    `ListServiceAccountKey` responses.  Google never exposes system-managed
    private keys, and never retains user-managed private keys.

    Values:
      TYPE_UNSPECIFIED: Unspecified. Equivalent to
        `TYPE_GOOGLE_CREDENTIALS_FILE`.
      TYPE_PKCS12_FILE: PKCS12 format. The password for the PKCS12 file is
        `notasecret`. For more information, see
        https://tools.ietf.org/html/rfc7292.
      TYPE_GOOGLE_CREDENTIALS_FILE: Google Credentials File format.
    """
        TYPE_UNSPECIFIED = 0
        TYPE_PKCS12_FILE = 1
        TYPE_GOOGLE_CREDENTIALS_FILE = 2
    name = _messages.StringField(1)
    privateKeyData = _messages.BytesField(2)
    privateKeyType = _messages.EnumField('PrivateKeyTypeValueValuesEnum', 3)
    publicKeyData = _messages.BytesField(4)
    validAfterTime = _messages.StringField(5)
    validBeforeTime = _messages.StringField(6)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CreateServiceAccountKeyRequest(_messages.Message):
    """The service account key create request.

  Enums:
    PrivateKeyTypeValueValuesEnum: The output format of the private key.
      `GOOGLE_CREDENTIALS_FILE` is the default output format.

  Fields:
    privateKeyType: The output format of the private key.
      `GOOGLE_CREDENTIALS_FILE` is the default output format.
  """

    class PrivateKeyTypeValueValuesEnum(_messages.Enum):
        """The output format of the private key. `GOOGLE_CREDENTIALS_FILE` is the
    default output format.

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
    privateKeyType = _messages.EnumField('PrivateKeyTypeValueValuesEnum', 1)
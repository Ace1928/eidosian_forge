from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyData(_messages.Message):
    """Represents a public key data along with its format.

  Enums:
    FormatValueValuesEnum: Output only. The format of the key.
    KeySpecValueValuesEnum: Required. The specifications for the key.

  Fields:
    format: Output only. The format of the key.
    key: Output only. The key data. The format of the key is represented by
      the format field.
    keySpec: Required. The specifications for the key.
    notAfterTime: Output only. Latest timestamp when this key is valid.
      Attempts to use this key after this time will fail. Only present if the
      key data represents a X.509 certificate.
    notBeforeTime: Output only. Earliest timestamp when this key is valid.
      Attempts to use this key before this time will fail. Only present if the
      key data represents a X.509 certificate.
  """

    class FormatValueValuesEnum(_messages.Enum):
        """Output only. The format of the key.

    Values:
      KEY_FORMAT_UNSPECIFIED: No format has been specified. This is an invalid
        format and must not be used.
      RSA_X509_PEM: A RSA public key wrapped in an X.509v3 certificate
        ([RFC5280] ( https://www.ietf.org/rfc/rfc5280.txt)), encoded in
        base64, and wrapped in [public certificate
        label](https://datatracker.ietf.org/doc/html/rfc7468#section-5.1).
    """
        KEY_FORMAT_UNSPECIFIED = 0
        RSA_X509_PEM = 1

    class KeySpecValueValuesEnum(_messages.Enum):
        """Required. The specifications for the key.

    Values:
      KEY_SPEC_UNSPECIFIED: No key specification specified.
      RSA_2048: A 2048 bit RSA key.
      RSA_3072: A 3072 bit RSA key.
      RSA_4096: A 4096 bit RSA key.
    """
        KEY_SPEC_UNSPECIFIED = 0
        RSA_2048 = 1
        RSA_3072 = 2
        RSA_4096 = 3
    format = _messages.EnumField('FormatValueValuesEnum', 1)
    key = _messages.StringField(2)
    keySpec = _messages.EnumField('KeySpecValueValuesEnum', 3)
    notAfterTime = _messages.StringField(4)
    notBeforeTime = _messages.StringField(5)
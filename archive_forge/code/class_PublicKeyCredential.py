from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublicKeyCredential(_messages.Message):
    """A public key format and data.

  Enums:
    FormatValueValuesEnum: The format of the key.

  Fields:
    format: The format of the key.
    key: The key data.
  """

    class FormatValueValuesEnum(_messages.Enum):
        """The format of the key.

    Values:
      UNSPECIFIED_PUBLIC_KEY_FORMAT: The format has not been specified. This
        is an invalid default value and must not be used.
      RSA_PEM: An RSA public key encoded in base64, and wrapped by `-----BEGIN
        PUBLIC KEY-----` and `-----END PUBLIC KEY-----`. This can be used to
        verify `RS256` signatures in JWT tokens ([RFC7518](
        https://www.ietf.org/rfc/rfc7518.txt)).
      RSA_X509_PEM: As RSA_PEM, but wrapped in an X.509v3 certificate
        ([RFC5280]( https://www.ietf.org/rfc/rfc5280.txt)), encoded in base64,
        and wrapped by `-----BEGIN CERTIFICATE-----` and `-----END
        CERTIFICATE-----`.
      ES256_PEM: Public key for the ECDSA algorithm using P-256 and SHA-256,
        encoded in base64, and wrapped by `-----BEGIN PUBLIC KEY-----` and
        `-----END PUBLIC KEY-----`. This can be used to verify JWT tokens with
        the `ES256` algorithm
        ([RFC7518](https://www.ietf.org/rfc/rfc7518.txt)). This curve is
        defined in [OpenSSL](https://www.openssl.org/) as the `prime256v1`
        curve.
      ES256_X509_PEM: As ES256_PEM, but wrapped in an X.509v3 certificate
        ([RFC5280]( https://www.ietf.org/rfc/rfc5280.txt)), encoded in base64,
        and wrapped by `-----BEGIN CERTIFICATE-----` and `-----END
        CERTIFICATE-----`.
    """
        UNSPECIFIED_PUBLIC_KEY_FORMAT = 0
        RSA_PEM = 1
        RSA_X509_PEM = 2
        ES256_PEM = 3
        ES256_X509_PEM = 4
    format = _messages.EnumField('FormatValueValuesEnum', 1)
    key = _messages.StringField(2)
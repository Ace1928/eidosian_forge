from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsValidationContext(_messages.Message):
    """[Deprecated] Defines the mechanism to obtain the Certificate Authority
  certificate to validate the client/server certificate. validate the
  client/server certificate.

  Enums:
    ValidationSourceValueValuesEnum: Defines how TLS certificates are
      obtained.

  Fields:
    certificatePath: The path to the file holding the CA certificate to
      validate the client or server certificate.
    sdsConfig: Specifies the config to retrieve certificates through SDS. This
      field is applicable only if tlsCertificateSource is set to USE_SDS.
    validationSource: Defines how TLS certificates are obtained.
  """

    class ValidationSourceValueValuesEnum(_messages.Enum):
        """Defines how TLS certificates are obtained.

    Values:
      INVALID: <no description>
      USE_PATH: USE_PATH specifies that the certificates and private key are
        obtained from a locally mounted filesystem path.
      USE_SDS: USE_SDS specifies that the certificates and private key are
        obtained from a SDS server.
    """
        INVALID = 0
        USE_PATH = 1
        USE_SDS = 2
    certificatePath = _messages.StringField(1)
    sdsConfig = _messages.MessageField('SdsConfig', 2)
    validationSource = _messages.EnumField('ValidationSourceValueValuesEnum', 3)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslConfigTemplate(_messages.Message):
    """Ssl config details of a connector version

  Enums:
    ClientCertTypeValueListEntryValuesEnum:
    ServerCertTypeValueListEntryValuesEnum:
    SslTypeValueValuesEnum: Controls the ssl type for the given connector
      version

  Fields:
    additionalVariables: Any additional fields that need to be rendered
    clientCertType: List of supported Client Cert Types
    isTlsMandatory: Boolean for determining if the connector version mandates
      TLS.
    serverCertType: List of supported Server Cert Types
    sslType: Controls the ssl type for the given connector version
  """

    class ClientCertTypeValueListEntryValuesEnum(_messages.Enum):
        """ClientCertTypeValueListEntryValuesEnum enum type.

    Values:
      CERT_TYPE_UNSPECIFIED: Cert type unspecified.
      PEM: Privacy Enhanced Mail (PEM) Type
    """
        CERT_TYPE_UNSPECIFIED = 0
        PEM = 1

    class ServerCertTypeValueListEntryValuesEnum(_messages.Enum):
        """ServerCertTypeValueListEntryValuesEnum enum type.

    Values:
      CERT_TYPE_UNSPECIFIED: Cert type unspecified.
      PEM: Privacy Enhanced Mail (PEM) Type
    """
        CERT_TYPE_UNSPECIFIED = 0
        PEM = 1

    class SslTypeValueValuesEnum(_messages.Enum):
        """Controls the ssl type for the given connector version

    Values:
      SSL_TYPE_UNSPECIFIED: No SSL configuration required.
      TLS: TLS Handshake
      MTLS: mutual TLS (MTLS) Handshake
    """
        SSL_TYPE_UNSPECIFIED = 0
        TLS = 1
        MTLS = 2
    additionalVariables = _messages.MessageField('ConfigVariableTemplate', 1, repeated=True)
    clientCertType = _messages.EnumField('ClientCertTypeValueListEntryValuesEnum', 2, repeated=True)
    isTlsMandatory = _messages.BooleanField(3)
    serverCertType = _messages.EnumField('ServerCertTypeValueListEntryValuesEnum', 4, repeated=True)
    sslType = _messages.EnumField('SslTypeValueValuesEnum', 5)
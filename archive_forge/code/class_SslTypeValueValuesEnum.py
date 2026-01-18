from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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
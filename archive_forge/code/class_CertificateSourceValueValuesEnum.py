from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateSourceValueValuesEnum(_messages.Enum):
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
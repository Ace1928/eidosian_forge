from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientCertTypeValueValuesEnum(_messages.Enum):
    """Type of Client Cert (PEM/JKS/.. etc.)

    Values:
      CERT_TYPE_UNSPECIFIED: Cert type unspecified.
      PEM: Privacy Enhanced Mail (PEM) Type
    """
    CERT_TYPE_UNSPECIFIED = 0
    PEM = 1
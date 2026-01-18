from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SignatureTypeValueValuesEnum(_messages.Enum):
    """Describes the type of resource associated with the signature.

    Values:
      SIGNATURE_TYPE_UNSPECIFIED: The default signature type.
      SIGNATURE_TYPE_PROCESS: Used for signatures concerning processes.
      SIGNATURE_TYPE_FILE: Used for signatures concerning disks.
    """
    SIGNATURE_TYPE_UNSPECIFIED = 0
    SIGNATURE_TYPE_PROCESS = 1
    SIGNATURE_TYPE_FILE = 2
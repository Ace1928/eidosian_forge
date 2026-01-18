from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessModeValueValuesEnum(_messages.Enum):
    """Either READ_ONLY, for allowing only read requests on the exported
    directory, or READ_WRITE, for allowing both read and write requests. The
    default is READ_WRITE.

    Values:
      ACCESS_MODE_UNSPECIFIED: AccessMode not set.
      READ_ONLY: The client can only read the file share.
      READ_WRITE: The client can read and write the file share (default).
    """
    ACCESS_MODE_UNSPECIFIED = 0
    READ_ONLY = 1
    READ_WRITE = 2
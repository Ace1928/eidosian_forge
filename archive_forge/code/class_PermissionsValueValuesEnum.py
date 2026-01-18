from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PermissionsValueValuesEnum(_messages.Enum):
    """Export permissions.

    Values:
      PERMISSIONS_UNSPECIFIED: Unspecified value.
      READ_ONLY: Read-only permission.
      READ_WRITE: Read-write permission.
    """
    PERMISSIONS_UNSPECIFIED = 0
    READ_ONLY = 1
    READ_WRITE = 2
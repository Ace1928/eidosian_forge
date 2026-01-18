from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NfsExport(_messages.Message):
    """A NFS export entry.

  Enums:
    PermissionsValueValuesEnum: Export permissions.

  Fields:
    allowDev: Allow dev flag in NfsShare AllowedClientsRequest.
    allowSuid: Allow the setuid flag.
    cidr: A CIDR range.
    machineId: Either a single machine, identified by an ID, or a comma-
      separated list of machine IDs.
    networkId: Network to use to publish the export.
    noRootSquash: Disable root squashing, which is a feature of NFS. Root
      squash is a special mapping of the remote superuser (root) identity when
      using identity authentication.
    permissions: Export permissions.
  """

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
    allowDev = _messages.BooleanField(1)
    allowSuid = _messages.BooleanField(2)
    cidr = _messages.StringField(3)
    machineId = _messages.StringField(4)
    networkId = _messages.StringField(5)
    noRootSquash = _messages.BooleanField(6)
    permissions = _messages.EnumField('PermissionsValueValuesEnum', 7)
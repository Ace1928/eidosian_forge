from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SimpleExportPolicyRule(_messages.Message):
    """An export policy rule describing various export options.

  Enums:
    AccessTypeValueValuesEnum: Access type (ReadWrite, ReadOnly, None)

  Fields:
    accessType: Access type (ReadWrite, ReadOnly, None)
    allowedClients: Comma separated list of allowed clients IP addresses
    hasRootAccess: Whether Unix root access will be granted.
    kerberos5ReadOnly: If enabled (true) the rule defines a read only access
      for clients matching the 'allowedClients' specification. It enables nfs
      clients to mount using 'authentication' kerberos security mode.
    kerberos5ReadWrite: If enabled (true) the rule defines read and write
      access for clients matching the 'allowedClients' specification. It
      enables nfs clients to mount using 'authentication' kerberos security
      mode. The 'kerberos5ReadOnly' value be ignored if this is enabled.
    kerberos5iReadOnly: If enabled (true) the rule defines a read only access
      for clients matching the 'allowedClients' specification. It enables nfs
      clients to mount using 'integrity' kerberos security mode.
    kerberos5iReadWrite: If enabled (true) the rule defines read and write
      access for clients matching the 'allowedClients' specification. It
      enables nfs clients to mount using 'integrity' kerberos security mode.
      The 'kerberos5iReadOnly' value be ignored if this is enabled.
    kerberos5pReadOnly: If enabled (true) the rule defines a read only access
      for clients matching the 'allowedClients' specification. It enables nfs
      clients to mount using 'privacy' kerberos security mode.
    kerberos5pReadWrite: If enabled (true) the rule defines read and write
      access for clients matching the 'allowedClients' specification. It
      enables nfs clients to mount using 'privacy' kerberos security mode. The
      'kerberos5pReadOnly' value be ignored if this is enabled.
    nfsv3: NFS V3 protocol.
    nfsv4: NFS V4 protocol.
  """

    class AccessTypeValueValuesEnum(_messages.Enum):
        """Access type (ReadWrite, ReadOnly, None)

    Values:
      ACCESS_TYPE_UNSPECIFIED: Unspecified Access Type
      READ_ONLY: Read Only
      READ_WRITE: Read Write
      READ_NONE: None
    """
        ACCESS_TYPE_UNSPECIFIED = 0
        READ_ONLY = 1
        READ_WRITE = 2
        READ_NONE = 3
    accessType = _messages.EnumField('AccessTypeValueValuesEnum', 1)
    allowedClients = _messages.StringField(2)
    hasRootAccess = _messages.StringField(3)
    kerberos5ReadOnly = _messages.BooleanField(4)
    kerberos5ReadWrite = _messages.BooleanField(5)
    kerberos5iReadOnly = _messages.BooleanField(6)
    kerberos5iReadWrite = _messages.BooleanField(7)
    kerberos5pReadOnly = _messages.BooleanField(8)
    kerberos5pReadWrite = _messages.BooleanField(9)
    nfsv3 = _messages.BooleanField(10)
    nfsv4 = _messages.BooleanField(11)
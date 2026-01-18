from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryServicesConfig(_messages.Message):
    """Directory Services configuration for Kerberos-based authentication.

  Fields:
    managedActiveDirectory: Configuration for Managed Service for Microsoft
      Active Directory.
  """
    managedActiveDirectory = _messages.MessageField('ManagedActiveDirectoryConfig', 1)
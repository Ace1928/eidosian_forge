from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaDisks(_messages.Message):
    """Disks defines the disks that would be attached to the workers.

  Fields:
    dockerRootDisk: Optional. Specifies the configuration for the docker root
      disk to be attached. If not specified, RBE will default to the RBE
      managed docker root disk.
    localSsd: Optional. Specifies the number of local SSDs to be attached. If
      specified, local SSDs will be used as the working directory.
  """
    dockerRootDisk = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaDisksPersistentDisk', 1)
    localSsd = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaDisksLocalSSD', 2)
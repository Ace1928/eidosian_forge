from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NfsMount(_messages.Message):
    """Represents a mount configuration for Network File System (NFS) to mount.

  Fields:
    mountPoint: Required. Destination mount path. The NFS will be mounted for
      the user under /mnt/nfs/
    path: Required. Source path exported from NFS server. Has to start with
      '/', and combined with the ip address, it indicates the source mount
      path in the form of `server:path`
    server: Required. IP address of the NFS server.
  """
    mountPoint = _messages.StringField(1)
    path = _messages.StringField(2)
    server = _messages.StringField(3)
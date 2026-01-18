from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2NFSVolumeSource(_messages.Message):
    """Represents an NFS mount.

  Fields:
    path: Path that is exported by the NFS server.
    readOnly: If true, the volume will be mounted as read only for all mounts.
    server: Hostname or IP address of the NFS server
  """
    path = _messages.StringField(1)
    readOnly = _messages.BooleanField(2)
    server = _messages.StringField(3)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Volume(_messages.Message):
    """Volume represents a named volume in a container.

  Fields:
    configMap: Not supported in Cloud Run.
    csi: Volume specified by the Container Storage Interface driver
    emptyDir: Ephemeral storage used as a shared volume.
    name: Volume's name. In Cloud Run Fully Managed, the name 'cloudsql' is
      reserved.
    nfs: A NFSVolumeSource attribute.
    secret: The secret's value will be presented as the content of a file
      whose name is defined in the item path. If no items are defined, the
      name of the file is the secretName.
  """
    configMap = _messages.MessageField('ConfigMapVolumeSource', 1)
    csi = _messages.MessageField('CSIVolumeSource', 2)
    emptyDir = _messages.MessageField('EmptyDirVolumeSource', 3)
    name = _messages.StringField(4)
    nfs = _messages.MessageField('NFSVolumeSource', 5)
    secret = _messages.MessageField('SecretVolumeSource', 6)
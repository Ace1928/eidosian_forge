from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VolumeMount(_messages.Message):
    """VolumeMount describes a mounting of a Volume within a container.

  Fields:
    mountPath: Required. Path within the container at which the volume should
      be mounted. Must not contain ':'.
    name: Required. The name of the volume. There must be a corresponding
      Volume with the same name.
    readOnly: Sets the mount to be read-only or read-write. Not used by Cloud
      Run.
    subPath: Path within the volume from which the container's volume should
      be mounted. Defaults to "" (volume's root).
  """
    mountPath = _messages.StringField(1)
    name = _messages.StringField(2)
    readOnly = _messages.BooleanField(3)
    subPath = _messages.StringField(4)
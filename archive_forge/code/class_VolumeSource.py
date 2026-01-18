from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VolumeSource(_messages.Message):
    """Volumes available to mount.

  Fields:
    emptyDir: A temporary directory that shares a pod's lifetime.
    name: Name of the Volume. Must be a DNS_LABEL and unique within the pod.
      More info: https://kubernetes.io/docs/concepts/overview/working-with-
      objects/names/#names
  """
    emptyDir = _messages.MessageField('EmptyDirVolumeSource', 1)
    name = _messages.StringField(2)
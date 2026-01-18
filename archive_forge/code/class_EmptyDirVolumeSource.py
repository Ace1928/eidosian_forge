from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EmptyDirVolumeSource(_messages.Message):
    """In memory (tmpfs) ephemeral storage. It is ephemeral in the sense that
  when the sandbox is taken down, the data is destroyed with it (it does not
  persist across sandbox runs).

  Fields:
    medium: The medium on which the data is stored. The default is "" which
      means to use the node's default medium. Must be an empty string
      (default) or Memory. More info:
      https://kubernetes.io/docs/concepts/storage/volumes#emptydir
    sizeLimit: Limit on the storage usable by this EmptyDir volume. The size
      limit is also applicable for memory medium. The maximum usage on memory
      medium EmptyDir would be the minimum value between the SizeLimit
      specified here and the sum of memory limits of all containers. The
      default is nil which means that the limit is undefined. More info:
      https://cloud.google.com/run/docs/configuring/in-memory-
      volumes#configure-volume. Info in Kubernetes:
      https://kubernetes.io/docs/concepts/storage/volumes/#emptydir
  """
    medium = _messages.StringField(1)
    sizeLimit = _messages.StringField(2)
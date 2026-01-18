from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PersistentDisk(_messages.Message):
    """Configuration for a persistent disk to be attached to the VM. See
  https://cloud.google.com/compute/docs/disks/performance for more information
  about disk type, size, and performance considerations.

  Fields:
    sizeGb: The size, in GB, of the disk to attach. If the size is not
      specified, a default is chosen to ensure reasonable I/O performance. If
      the disk type is specified as `local-ssd`, multiple local drives are
      automatically combined to provide the requested size. Note, however,
      that each physical SSD is 375GB in size, and no more than 8 drives can
      be attached to a single instance.
    sourceImage: An image to put on the disk before attaching it to the VM.
    type: The Compute Engine disk type. If unspecified, `pd-standard` is used.
  """
    sizeGb = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    sourceImage = _messages.StringField(2)
    type = _messages.StringField(3)
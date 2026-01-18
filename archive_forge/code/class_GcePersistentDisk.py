from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcePersistentDisk(_messages.Message):
    """An EphemeralDirectory is backed by a Compute Engine persistent disk.

  Fields:
    diskType: Optional. Type of the disk to use. Defaults to `"pd-standard"`.
    readOnly: Optional. Whether the disk is read only. If true, the disk may
      be shared by multiple VMs and source_snapshot must be set.
    sourceImage: Optional. Name of the disk image to use as the source for the
      disk. Must be empty if source_snapshot is set. Updating source_image
      will update content in the ephemeral directory after the workstation is
      restarted. This field is mutable.
    sourceSnapshot: Optional. Name of the snapshot to use as the source for
      the disk. Must be empty if source_image is set. Must be empty if
      read_only is false. Updating source_snapshot will update content in the
      ephemeral directory after the workstation is restarted. This field is
      mutable.
  """
    diskType = _messages.StringField(1)
    readOnly = _messages.BooleanField(2)
    sourceImage = _messages.StringField(3)
    sourceSnapshot = _messages.StringField(4)
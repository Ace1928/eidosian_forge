from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaDisksPersistentDisk(_messages.Message):
    """PersistentDisk specifies how to attach a persistent disk to the workers.

  Fields:
    diskSizeGb: Required. Size of the disk in GB.
    diskType: Required. Type of disk attached (supported types are pd-standard
      and pd-ssd).
    sourceImage: Required. VM image to use for the disk.
  """
    diskSizeGb = _messages.IntegerField(1)
    diskType = _messages.StringField(2)
    sourceImage = _messages.StringField(3)
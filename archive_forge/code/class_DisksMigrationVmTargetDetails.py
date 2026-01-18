from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisksMigrationVmTargetDetails(_messages.Message):
    """Details for the VM created VM as part of disks migration.

  Fields:
    vmUri: Output only. The URI of the Compute Engine VM.
  """
    vmUri = _messages.StringField(1)
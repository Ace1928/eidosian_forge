from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataDiskAssignment(_messages.Message):
    """Data disk assignment for a given VM instance.

  Fields:
    dataDisks: Mounted data disks. The order is important a data disk's
      0-based index in this list defines which persistent directory the disk
      is mounted to, for example the list of {
      "myproject-1014-104817-4c2-harness-0-disk-0" }, {
      "myproject-1014-104817-4c2-harness-0-disk-1" }.
    vmInstance: VM instance name the data disks mounted to, for example
      "myproject-1014-104817-4c2-harness-0".
  """
    dataDisks = _messages.StringField(1, repeated=True)
    vmInstance = _messages.StringField(2)
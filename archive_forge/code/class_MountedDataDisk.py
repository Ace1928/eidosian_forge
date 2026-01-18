from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MountedDataDisk(_messages.Message):
    """Describes mounted data disk.

  Fields:
    dataDisk: The name of the data disk. This name is local to the Google
      Cloud Platform project and uniquely identifies the disk within that
      project, for example "myproject-1014-104817-4c2-harness-0-disk-1".
  """
    dataDisk = _messages.StringField(1)
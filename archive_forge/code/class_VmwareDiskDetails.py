from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareDiskDetails(_messages.Message):
    """The details of a Vmware VM disk.

  Fields:
    diskNumber: The ordinal number of the disk.
    label: The disk label.
    sizeGb: Size in GB.
  """
    diskNumber = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    label = _messages.StringField(2)
    sizeGb = _messages.IntegerField(3)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResizeDiskRequest(_messages.Message):
    """Request for resizing the notebook instance disks

  Fields:
    bootDisk: Required. The boot disk to be resized. Only disk_size_gb will be
      used.
    dataDisk: Required. The data disk to be resized. Only disk_size_gb will be
      used.
  """
    bootDisk = _messages.MessageField('BootDisk', 1)
    dataDisk = _messages.MessageField('DataDisk', 2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResizeVolumeRequest(_messages.Message):
    """Request for emergency resize Volume.

  Fields:
    sizeGib: New Volume size, in GiB.
  """
    sizeGib = _messages.IntegerField(1)
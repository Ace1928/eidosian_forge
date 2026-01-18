from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsFuseCsiDriverConfig(_messages.Message):
    """Configuration for the Cloud Storage Fuse CSI driver.

  Fields:
    enabled: Whether the Cloud Storage Fuse CSI driver is enabled for this
      cluster.
  """
    enabled = _messages.BooleanField(1)
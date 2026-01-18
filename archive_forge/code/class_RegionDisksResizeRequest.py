from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionDisksResizeRequest(_messages.Message):
    """A RegionDisksResizeRequest object.

  Fields:
    sizeGb: The new size of the regional persistent disk, which is specified
      in GB.
  """
    sizeGb = _messages.IntegerField(1)
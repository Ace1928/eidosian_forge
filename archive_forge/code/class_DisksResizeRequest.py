from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisksResizeRequest(_messages.Message):
    """A DisksResizeRequest object.

  Fields:
    sizeGb: The new size of the persistent disk, which is specified in GB.
  """
    sizeGb = _messages.IntegerField(1)
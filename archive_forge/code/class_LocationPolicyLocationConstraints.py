from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocationPolicyLocationConstraints(_messages.Message):
    """Per-zone constraints on location policy for this zone.

  Fields:
    maxCount: Maximum number of items that are allowed to be placed in this
      zone. The value must be non-negative.
  """
    maxCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
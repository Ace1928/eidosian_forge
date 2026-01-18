from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AcceleratorsValueListEntry(_messages.Message):
    """A AcceleratorsValueListEntry object.

    Fields:
      guestAcceleratorCount: Number of accelerator cards exposed to the guest.
      guestAcceleratorType: The accelerator type resource name, not a full
        URL, e.g. nvidia-tesla-t4.
    """
    guestAcceleratorCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    guestAcceleratorType = _messages.StringField(2)
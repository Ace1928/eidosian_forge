from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OwnerTypeValueValuesEnum(_messages.Enum):
    """Output only. Whether the device is owned by the company or an
    individual

    Values:
      DEVICE_OWNERSHIP_UNSPECIFIED: Default value. The value is unused.
      COMPANY: Company owns the device.
      BYOD: Bring Your Own Device (i.e. individual owns the device)
    """
    DEVICE_OWNERSHIP_UNSPECIFIED = 0
    COMPANY = 1
    BYOD = 2